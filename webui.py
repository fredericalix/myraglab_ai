import streamlit as st
from script import LocalRAG
import time

# Configuration de la page
st.set_page_config(
    page_title="Lempire Infrastructure RAG",
    page_icon="🤖",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e9ecef;
}
.bot-message {
    background-color: #f8f9fa;
}
.source-info {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'rag' not in st.session_state:
    st.session_state.rag = LocalRAG()
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Titre de l'application
st.title("🤖 Assistant Documentation Infrastructure")
st.markdown("---")

# Zone de chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>Vous:</b> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <b>Assistant:</b> {message["response"]}
            <div class="source-info">
                <b>Sources:</b><br>
                {message["sources"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Zone de saisie
with st.container():
    question = st.text_input("Votre question:", key="user_input")
    
    if st.button("Envoyer", key="send_button"):
        if question:
            # Ajout de la question à l'historique
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            # Affichage d'un spinner pendant la recherche
            with st.spinner("Recherche en cours..."):
                try:
                    # Recherche des documents pertinents
                    results = st.session_state.rag.vector_search(question, top_k=3)
                    
                    if not results:
                        response = "❌ Aucun document pertinent trouvé."
                        sources = "Aucune source disponible"
                    else:
                        # Construction du contexte
                        context = "\n\n".join([
                            f"Document '{r['title']}' ({r['category']}):\n{r['content'][:500]}..."
                            for r in results
                        ])
                        
                        # Construction du prompt
                        prompt = f"""Tu es un assistant expert en infrastructure qui aide à comprendre la documentation de Lempire.
                        Utilise uniquement les informations du contexte ci-dessous pour répondre à la question.
                        Si tu ne peux pas répondre avec le contexte donné, dis-le clairement.
                        
                        Contexte:
                        {context}
                        
                        Question: {question}
                        
                        Réponse:"""
                        
                        # Appel à l'API de LM Studio
                        import requests
                        response = requests.post(
                            f"{st.session_state.rag.llm_url}/chat/completions",
                            json={
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0.7,
                                "max_tokens": 1000
                            }
                        )
                        
                        if response.status_code == 200:
                            response = response.json()['choices'][0]['message']['content']
                            sources = "\n".join([
                                f"- {r['title']} ({r['category']}) - Similarité: {r['similarity']:.2%}"
                                for r in results
                            ])
                        else:
                            response = f"❌ Erreur lors de l'appel à LM Studio: {response.status_code}"
                            sources = "Erreur lors de la récupération des sources"
                    
                    # Ajout de la réponse à l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": question,
                        "response": response,
                        "sources": sources
                    })
                    
                    # Recharge la page pour afficher la nouvelle réponse
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

# Bouton pour effacer l'historique
if st.button("Effacer l'historique"):
    st.session_state.messages = []
    st.rerun() 