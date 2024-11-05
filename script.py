import numpy as np
from redis import Redis, ResponseError
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.query import Query
import argparse
import os
from pathlib import Path
import hashlib
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import json
import requests

def verify_redis_stack():
    """Vérifie si Redis Stack est installé avec les capacités vectorielles"""
    try:
        # Création d'une connexion temporaire
        temp_client = Redis(host='localhost', port=6379)
        
        # Vérification des modules chargés
        modules = temp_client.module_list()
        
        # Recherche du module RediSearch
        has_search = any(
            module[b'name'].decode('utf-8') == 'search'
            for module in modules
        )
        
        if not has_search:
            print("❌ Module RediSearch non trouvé")
            return False
            
        print("✅ Redis Stack vérifié avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de Redis Stack: {e}")
        return False

class LocalRAG:
    def __init__(self, 
                 redis_host='localhost', 
                 redis_port=6379, 
                 index_name='docs',
                 llm_url='http://localhost:1234/v1'):
                 
        # Vérification des capacités vectorielles
        if not verify_redis_stack():
            raise RuntimeError(
                "Redis Stack n'est pas correctement installé. "
                "Les capacités vectorielles ne sont pas disponibles."
            )
            
        self.redis_client = Redis(host=redis_host, port=redis_port)
        self.index_name = index_name
        self.llm_url = llm_url
        self.vector_dim = 4096
        
        # Test de connexion à LM Studio
        try:
            response = requests.get(f"{self.llm_url}/models")
            if response.status_code == 200:
                print("✅ Connexion à LM Studio établie")
            else:
                print("❌ Impossible de se connecter à LM Studio")
        except Exception as e:
            print(f"❌ Erreur de connexion à LM Studio: {e}")
    
    def get_embedding(self, text: str) -> list:
        """Obtient l'embedding d'un texte via LM Studio"""
        try:
            print("    📡 Appel à l'API d'embedding...")
            
            # Tronquer le texte si trop long (par exemple, limiter à 8000 caractères)
            text = text[:8000] if len(text) > 8000 else text
            
            response = requests.post(
                f"{self.llm_url}/embeddings",
                json={
                    "input": text,
                    "model": "local"
                },
                timeout=30  # Timeout de 30 secondes
            )
            
            if response.status_code == 200:
                embedding = response.json()['data'][0]['embedding']
                print(f"    ✅ Embedding reçu ({len(embedding)} dimensions)")
                return embedding
            else:
                print(f"    ❌ Erreur HTTP: {response.status_code}")
                print(f"    📝 Réponse: {response.text}")
                return None
            
        except requests.Timeout:
            print("    ❌ Timeout lors de l'appel à l'API d'embedding")
            return None
        except Exception as e:
            print(f"    ❌ Erreur lors de l'appel à l'API d'embedding: {e}")
            return None
    
    def _create_vector_index(self):
        """Crée l'index vectoriel dans Redis"""
        try:
            # Suppression de l'ancien index s'il existe
            try:
                self.redis_client.ft(self.index_name).dropindex(delete_docs=False)
                print("✅ Ancien index supprimé")
            except:
                pass
                
            # Création du schéma
            schema = (
                TextField("content"),
                TextField("title"),
                TagField("source"),
                TagField("category"),
                TextField("full_path"),
                VectorField("embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 768,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 40,
                        "EF_CONSTRUCTION": 200,
                        "EF_RUNTIME": 10,
                        "INITIAL_CAP": 100
                    }
                )
            )
            
            # Création de l'index
            self.redis_client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[f"{self.index_name}:"])
            )
            print("✅ Index vectoriel créé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de l'index: {e}")
            raise
    
    def load_documents(self, docs_path: str) -> None:
        """Charge les documents depuis le chemin spécifié dans Redis"""
        # D'abord, supprimer tous les documents existants
        print("🗑️ Nettoyage de la base...")
        for key in self.redis_client.scan_iter(f"{self.index_name}:*"):
            self.redis_client.delete(key)
        
        # Ensuite, créer l'index
        print("📑 Création de l'index vectoriel...")
        self._create_vector_index()
        
        docs_path = Path(docs_path)
        if not docs_path.exists():
            raise ValueError(f"Le chemin {docs_path} n'existe pas")
        
        print(f"\n📂 Chargement des documents depuis {docs_path}")
        
        # Extensions supportées
        supported_extensions = {'.md', '.txt', '.rst', '.yaml', '.yml'}
        files_processed = 0
        
        # Lister tous les fichiers avant de commencer
        files = [f for f in docs_path.rglob('*') if f.suffix.lower() in supported_extensions]
        total_files = len(files)
        print(f"📁 {total_files} fichiers trouvés à traiter")
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{total_files}] Traitement de {file_path.name}...")
            try:
                content = file_path.read_text(encoding='utf-8')
                embedding = self.get_embedding(content)
                if embedding is None:
                    raise ValueError("Impossible de générer l'embedding")
                
                doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
                
                # Stockage dans Redis avec le préfixe correct
                self.redis_client.hset(
                    f"{self.index_name}:{doc_id}",
                    mapping={
                        'content': content,
                        'title': file_path.stem,
                        'source': file_path.suffix[1:],
                        'category': file_path.parent.name,
                        'full_path': str(file_path),
                        'embedding': np.array(embedding, dtype=np.float32).tobytes()  # Stockage direct en bytes
                    }
                )
                
                files_processed += 1
                print(f"✅ {file_path.name} chargé avec succès")
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {file_path}: {e}")
                continue
        
        print(f"\n📊 Total: {files_processed}/{total_files} documents chargés")
    
    def vector_search(self, query: str, top_k=3, category=None):
        """Recherche vectorielle"""
        try:
            print("\n🔍 Recherche en cours...")
            query_vector = self.get_embedding(query)
            if query_vector is None:
                raise ValueError("Impossible de générer l'embedding de la requête")
            
            print(f"✅ Embedding généré (dimension: {len(query_vector)})")
            
            base_query = f"*=>[KNN {top_k} @embedding $query_vector AS score]"
            if category:
                base_query = f"@category:{{{category}}} {base_query}"
            
            print(f"🔎 Requête Redis: {base_query}")
            
            query = Query(base_query)\
                .return_fields("content", "title", "source", "category", "full_path", "score")\
                .dialect(2)\
                .sort_by("score")\
                .paging(0, top_k)
            
            params_dict = {
                "query_vector": np.array(query_vector, dtype=np.float32).tobytes()
            }
            
            results = self.redis_client.ft(self.index_name).search(query, params_dict)
            print(f"📊 Nombre de résultats: {len(results.docs)}")
            
            return [{
                "content": doc.content,
                "title": doc.title,
                "source": doc.source,
                "category": doc.category,
                "full_path": doc.full_path,
                "similarity": 1 - float(doc.score)
            } for doc in results.docs]
            
        except Exception as e:
            print(f"❌ Erreur lors de la recherche vectorielle : {e}")
            raise
    
    def check_database(self):
        """Affiche un rapport complet de la base de données"""
        print("\n" + "="*50)
        print("📊 RAPPORT DE LA BASE DE DONNÉES REDIS")
        print("="*50)

        # Statistiques de l'index
        try:
            info = self.redis_client.ft(self.index_name).info()
            print("\n📈 STATISTIQUES GÉNÉRALES:")
            print(f"- Nombre total de documents: {info['num_docs']}")
            print(f"- Taille de l'index: {info.get('inverted_sz_mb', '0')} MB")
            print(f"- Nombre de termes indexés: {info['num_terms']}")
        except Exception as e:
            print(f"❌ Erreur lors de la lecture des statistiques: {e}")

        # Liste des documents
        try:
            query = Query("*").dialect(2)
            results = self.redis_client.ft(self.index_name).search(query)
            
            print("\n📚 DOCUMENTS STOCKÉS:")
            for i, doc in enumerate(results.docs, 1):
                print(f"\nDocument {i}:")
                print(f"  📄 Titre: {doc.title}")
                print(f"  🏷️ Catégorie: {doc.category}")
                print(f"  📂 Source: {doc.source}")
                print(f"   Chemin: {doc.full_path}")
                print(f"  📝 Aperçu: {doc.content[:100]}...")
        except Exception as e:
            print(f"❌ Erreur lors de la lecture des documents: {e}")

        print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Gestionnaire de base de données Redis RAG")
    parser.add_argument("-check", action="store_true", 
                       help="Affiche les informations de la base de données")
    parser.add_argument("--docs", type=str, 
                       help="Chemin vers le dossier contenant les documents à charger")
    parser.add_argument("--chat", action="store_true",
                       help="Démarre une session de chat interactive avec le RAG")
    
    args = parser.parse_args()
    
    rag = LocalRAG()
    
    if args.docs:
        rag.load_documents(args.docs)
    elif args.check:
        rag.check_database()
    elif args.chat:
        print("\n🤖 Bienvenue dans le chat RAG! (tapez 'quit' pour quitter)")
        print("📚 Contexte: documentation infrastructure Lempire")
        
        while True:
            try:
                question = input("\n❓ Votre question: ")
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                    
                # Recherche des documents pertinents
                results = rag.vector_search(question, top_k=3)
                
                if not results:
                    print("❌ Aucun document pertinent trouvé.")
                    continue
                
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
                response = requests.post(
                    f"{rag.llm_url}/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    }
                )
                
                if response.status_code == 200:
                    answer = response.json()['choices'][0]['message']['content']
                    print("\n🤖 Réponse:")
                    print(answer)
                    
                    print("\n📚 Sources utilisées:")
                    for r in results:
                        print(f"- {r['title']} ({r['category']}) - Similarité: {r['similarity']:.2%}")
                else:
                    print(f"❌ Erreur lors de l'appel à LM Studio: {response.status_code}")
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
