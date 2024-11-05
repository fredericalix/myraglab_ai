RAG Documentation Infrastructure
==============================
Un outil de RAG (Retrieval-Augmented Generation) pour interroger la documentation d'infrastructure via une interface en ligne de commande ou une interface web.
INSTALLATION
-----------
Prérequis :
Python 3.11+
Redis Stack
LM Studio (avec un modèle local configuré)
Installation de Redis Stack
----------------------------
MacOS :
brew install redis-stack
Ubuntu :
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server
Installation de l'environnement Python
--------------------------------------
Création de l'environnement virtuel :
python -m venv venv
Activation de l'environnement :
Unix/MacOS : source venv/bin/activate
Windows : venv\Scripts\activate
Installation des dépendances :
pip install -r requirements.txt
Installation de LM Studio
-------------------------
Téléchargez LM Studio depuis https://lmstudio.ai/
Installez et lancez l'application
Téléchargez le modèle Meta-Llama-3.1-8B-Instruct-GGUF
Activez le serveur local dans LM Studio (bouton "Local Server")
UTILISATION
----------
Configuration initiale :
Vérifiez que Redis fonctionne :
redis-cli ping
(devrait répondre PONG)
Vérifiez LM Studio :
Modèle chargé
Serveur local actif
URL : http://localhost:1234/v1
Chargement des documents :
------------------------
Dossier complet :
python script.py --docs ~/Documents/documentation/
Fichier unique :
python script.py --docs ~/Documents/doc.pdf
Ajout sans effacer :
python script.py --docs ~/Documents/nouveau_doc.pdf --incremental
Interface en ligne de commande :
python script.py --chat
Interface Web :
streamlit run webui.py
Accessible sur : http://localhost:8501
Vérification de la base :
python script.py -check
TYPES DE FICHIERS SUPPORTÉS
-------------------------
Markdown (.md)
Texte (.txt)
ReStructuredText (.rst)
YAML (.yaml, .yml)
PDF (.pdf)
OPTIONS
-------
python script.py --help
Options disponibles :
--docs PATH : Chemin vers les documents
--incremental : Ajout sans effacer
--chat : Mode chat
-check : État de la base
DÉPANNAGE
---------
Redis ne répond pas :
brew services start redis-stack (MacOS)
sudo systemctl start redis-stack-server (Ubuntu)
LM Studio inaccessible :
Vérifier que LM Studio est lancé
Vérifier le serveur local
Vérifier l'URL
Vérifier le modèle
Index existe déjà :
Utiliser --incremental
Ou nettoyer : redis-cli FLUSHALL
ARCHITECTURE
-----------
Redis Stack : stockage et recherche
LM Studio : modèle de langage
Streamlit : interface web
Embeddings : générés localement
LICENCE
-------
MIT
CONTRIBUTION
-----------
Contributions bienvenues via issues ou pull requests.
RESSOURCES
----------
Documentation Redis Stack : https://redis.io/docs/stack/
Documentation Streamlit : https://docs.streamlit.io/
LM Studio : https://lmstudio.ai/
]