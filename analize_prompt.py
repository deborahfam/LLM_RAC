import re

def analyze_prompt(prompt: str) -> dict:
    """
    Analiza el prompt para decidir el método de búsqueda y el tamaño de k.
    
    Args:
        prompt (str): El prompt del usuario.
    
    Returns:
        dict: Un diccionario con el método de búsqueda y el tamaño de k.
    """
    # Heurísticas básicas para analizar el prompt
    length = len(prompt.split())
    question_words = re.findall(r'\b(what|who|when|where|why|how|which)\b', prompt.lower())
    
    # Decidir el método de búsqueda
    if len(question_words) > 0 and length < 10:
        search_method = "similarity"
        k = 3  # Pequeño k para respuestas precisas
    elif len(question_words) > 0:
        search_method = "similarity"
        k = 5  # Mediano k para respuestas precisas con más contexto
    else:
        search_method = "mmr"
        k = 5  # Grande k para respuestas diversas
    
    return {"search_method": search_method, "k": k}
