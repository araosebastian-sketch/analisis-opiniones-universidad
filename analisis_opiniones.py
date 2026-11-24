import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. EXPLORACIÓN INICIAL (ruta relativa)
df = pd.read_csv("dataset_opiniones.csv")

print("=== 1. EXPLORACIÓN INICIAL ===")
print(f"Total registros: {len(df)} | Columnas: {list(df.columns)}")
print("Datos vacíos por columna:\n", df.isnull().sum())
print("\nComentarios por área:\n", df["area"].value_counts())
print("\nComentarios por polaridad de referencia:\n", df["polaridad_referencia"].value_counts())
print("\nCruce Área vs. Polaridad:\n", pd.crosstab(df["area"], df["polaridad_referencia"], margins=True))

df["num_palabras"] = df["comentario"].apply(lambda x: len(str(x).split()))
print(f"\nPromedio de palabras por comentario: {df['num_palabras'].mean():.2f}")

# 2. LIMPIEZA Y LEMATIZACIÓN
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "ante", "bajo", "cabe", "con", "contra", "desde", "durante", "en",
    "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin",
    "sobre", "tras", "y", "e", "ni", "o", "u", "pero", "mas", "sino", "que",
    "cual", "quien", "cuyo", "donde", "cuando", "como", "cuanto", "se", "su",
    "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "nuestros", "nuestras",
    "lo", "le", "les", "me", "te", "nos", "os", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
    "es", "son", "fue", "fueron", "era", "eran", "ser", "estar", "está", "están",
    "estuvo", "hay", "ha", "han", "había", "muy", "más", "menos", "tan", "tanto",
    "cada", "todo", "toda", "todos", "todas", "otro", "otra", "otros", "otras",
    "mismo", "misma", "ya", "aún", "todavía", "así", "bien", "mal", "vez", "veces",
    "hace", "parte", "solo", "sólo", "tener", "tiene", "tienen", "tuve"
}

LEMAS_ES = {
    "profesores": "profesor", "docentes": "docente", "facilitadores": "facilitador",
    "explica": "explicar", "explican": "explicar", "ejemplos": "ejemplo",
    "prácticos": "práctico", "prácticas": "práctica", "cae": "caer",
    "exámenes": "examen", "finales": "final", "aires": "aire",
    "acondicionados": "acondicionado", "funcionan": "funcionar", "funciona": "funcionar",
    "funcionó": "funcionar", "semanas": "semana", "estudios": "estudio",
    "clases": "clase", "puntuales": "puntual", "cumple": "cumplir",
    "establecido": "establecer", "archivos": "archivo", "bases": "base",
    "datos": "dato", "científicas": "científico", "entiendo": "entender",
    "responde": "responder", "dudas": "duda", "equipos": "equipo",
    "actualizados": "actualizado", "contesta": "contestar", "correos": "correo",
    "días": "día", "esperando": "esperar", "evaluaciones": "evaluación",
    "excesivamente": "excesivo", "largas": "largo", "grabadas": "grabar",
    "queden": "quedar", "disponibles": "disponible", "baños": "baño",
    "necesitan": "necesitar", "motiva": "motivar", "investigar": "investigar",
    "problemas": "problema", "subir": "subir", "salas": "sala",
    "silenciosas": "silencioso", "cómodas": "cómodo", "trabajar": "trabajar",
    "sube": "subir", "diapositivas": "diapositiva", "áreas": "área",
    "comunes": "común", "organizar": "organizar", "inscripciones": "inscripción",
    "hacen": "hacer", "perder": "perder", "usar": "usar",
    "retroalimentaciones": "retroalimentación", "proyectos": "proyecto",
    "llegan": "llegar", "ayudan": "ayudar", "mejorar": "mejorar",
    "cuenta": "contar", "laboratorios": "laboratorio", "errores": "error",
    "notificaciones": "notificación", "tareas": "tarea", "faltan": "faltar",
    "enchufes": "enchufe", "salones": "salón", "conectar": "conectar",
    "laptops": "laptop", "extenderse": "extender", "intentar": "intentar",
    "descargar": "descargar", "guías": "guía", "fomenta": "fomentar",
    "sillas": "silla", "aulas": "aula", "deterioradas": "deteriorado",
    "incómodas": "incómodo", "atendieron": "atender", "demoras": "demora",
    "resolver": "resolver", "referencias": "referencia", "años": "año",
    "comunicarse": "comunicar", "compañeros": "compañero", "conferencias": "conferencia",
    "entrega": "entregar", "notas": "nota", "llegamos": "llegar",
    "saber": "saber", "solicité": "solicitar", "siguió": "seguir",
    "calificaciones": "calificación", "interrumpe": "interrumpir",
    "dinámicas": "dinámico", "participativas": "participativo", "casos": "caso",
    "reales": "real", "envían": "enviar", "permite": "permitir",
    "reservar": "reservar", "libros": "libro", "físicos": "físico",
    "cubrió": "cubrir", "precios": "precio", "altos": "alto",
    "alimentos": "alimento", "saludables": "saludable", "requisitos": "requisito",
    "pude": "poder", "presentar": "presentar", "quedó": "quedar",
    "cargando": "cargar", "infinitamente": "infinito", "utiliza": "utilizar",
    "herramientas": "herramienta", "carnets": "carnet", "estudiantiles": "estudiantil",
    "realizó": "realizar", "anunciada": "anunciar", "integrar": "integrar",
    "recordatorios": "recordatorio", "automáticos": "automático", "entregas": "entrega",
    "verdes": "verde", "espacios": "espacio", "recreativos": "recreativo",
    "instrucciones": "instrucción", "confusas": "confuso",
    "contradictorias": "contradictorio", "orientar": "orientar", "nuevos": "nuevo",
    "ingresos": "ingreso", "credenciales": "credencial", "institucionales": "institucional",
    "inconvenientes": "inconveniente", "realizar": "realizar", "redes": "red"
}

def limpiar_lematizar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\.\S+|https\S+", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    tokens = [
        LEMAS_ES.get(t, t)
        for t in texto.split()
        if t not in STOPWORDS_ES and len(t) > 2
    ]
    return " ".join(tokens)

df["texto_limpio"] = df["comentario"].apply(limpiar_lematizar)

print("\n=== 2. COMPARACIÓN DE 5 COMENTARIOS ===")
print(df.loc[[0, 1, 11, 19, 49], ["id", "comentario", "texto_limpio"]].to_string(index=False))

# 3. EXTRACCIÓN DE PALABRAS CLAVE (TF-IDF)
df_neg = df[df["polaridad_referencia"] == "Negativa"]
vec_neg = TfidfVectorizer(ngram_range=(1, 2), max_features=15)
X_neg = vec_neg.fit_transform(df_neg["texto_limpio"])
scores_neg = np.asarray(X_neg.mean(axis=0)).ravel()
terms_neg = vec_neg.get_feature_names_out()
df_top_neg = pd.DataFrame({"termino": terms_neg, "tfidf": scores_neg}).sort_values("tfidf", ascending=True)

plt.figure(figsize=(9, 5.2))
plt.barh(df_top_neg["termino"], df_top_neg["tfidf"], color="#c0392b")
plt.title("Top 15 términos y bigramas en comentarios negativos (TF-IDF)")
plt.xlabel("Puntaje TF-IDF promedio")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("top15_negativos_tfidf.png", dpi=150)
plt.close()

areas = df["area"].unique()
fig, axes = plt.subplots(nrows=len(areas), ncols=1, figsize=(8.5, 11))
colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

for ax, area, color in zip(axes, areas, colores):
    sub = df[df["area"] == area]["texto_limpio"]
    vec_a = TfidfVectorizer(ngram_range=(1, 2), max_features=6)
    X_a = vec_a.fit_transform(sub)
    sc_a = np.asarray(X_a.mean(axis=0)).ravel()
    tm_a = vec_a.get_feature_names_out()
    orden = np.argsort(sc_a)
    ax.barh(tm_a[orden], sc_a[orden], color=color)
    ax.set_title(f"Términos más relevantes - {area}")
    ax.set_xlabel("TF-IDF promedio")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig("terminos_por_area.png", dpi=150)
plt.close()

# 4. ANÁLISIS DE SENTIMIENTO POR REGLAS / LÉXICO
LEXICO_POS = {
    "claridad", "prácticos", "rápido", "amable", "excelente", "buen", "buena",
    "buenos", "actualizados", "gusta", "disponibles", "motiva", "silenciosas",
    "cómodas", "intuitiva", "fácil", "ayudan", "mejorar", "seguro",
    "fomenta", "resolver", "dinámicas", "participativas", "permite",
    "gran", "calidad", "enriquecer", "paciencia", "correctamente", "puntuales", "cumple"
}
LEXICO_NEG = {
    "cae", "frustrante", "funcionan", "falta", "faltan", "entiendo",
    "contesta", "excesivamente", "largas", "mantenimiento", "urgente",
    "problemas", "pésima", "inestable", "mal", "perder", "demasiada", "poca",
    "retraso", "error", "deterioradas", "incómodas", "desactualizado",
    "ruido", "interrumpe", "altos", "infinitamente", "apenas",
    "confusas", "contradictorias", "imposible"
}
NEGACIONES = {"no", "ni", "nunca", "jamás", "sin", "nadie"}

def clasificar_sentimiento(texto):
    palabras = re.findall(r"[a-záéíóúüñ]+", str(texto).lower())
    puntaje = 0
    for i, p in enumerate(palabras):
        ventana = palabras[max(0, i - 2):i]
        negado = any(w in NEGACIONES for w in ventana)
        if p in LEXICO_POS:
            puntaje += -1 if negado else 1
        elif p in LEXICO_NEG:
            puntaje += 1 if negado else -1
    if puntaje > 0:
        return "Positiva"
    elif puntaje < 0:
        return "Negativa"
    return "Neutra"

df["polaridad_predicha"] = df["comentario"].apply(clasificar_sentimiento)
acc = (df["polaridad_referencia"] == df["polaridad_predicha"]).mean() * 100

print("\n=== 4. ANÁLISIS DE SENTIMIENTO ===")
print(f"Exactitud: {acc:.2f}% ({(df['polaridad_referencia'] == df['polaridad_predicha']).sum()}/60)")
errores = df[df["polaridad_referencia"] != df["polaridad_predicha"]]
print("\nCasos con discrepancia:")
print(errores[["id", "area", "polaridad_referencia", "polaridad_predicha", "comentario"]].to_string(index=False))
