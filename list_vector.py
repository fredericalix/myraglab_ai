import redis

def list_vector_indexes():
    # Connexion à Redis (modifier les paramètres si nécessaire)
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Liste pour stocker les noms d'index contenant des champs VECTOR
    vector_indexes = []
    
    # Récupère tous les index
    try:
        indexes = r.execute_command("FT._LIST")
    except redis.exceptions.ResponseError:
        print("Assurez-vous que le module RedisSearch est bien activé.")
        return []

    # Parcourt chaque index et vérifie les champs VECTOR
    for index_name in indexes:
        info = r.execute_command(f"FT.INFO {index_name}")
        
        # Recherche des attributs et vérifie le type VECTOR
        attributes_index = info.index("attributes") + 1  # Position des attributs
        attributes = info[attributes_index]
        
        for attribute in attributes:
            if "VECTOR" in attribute:
                vector_indexes.append(index_name)
                break  # Passe au prochain index si un champ VECTOR est trouvé

    return vector_indexes

# Exécution et affichage des résultats
vector_indexes = list_vector_indexes()
if vector_indexes:
    print("Les index contenant des champs VECTOR sont :", vector_indexes)
else:
    print("Aucun index contenant de champ VECTOR n'a été trouvé.")

