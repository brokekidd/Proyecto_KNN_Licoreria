import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. CREDENCIALES DE CONEXIÓN A SUPABASE (POSTGRESQL)
# (Búscalas en Supabase: Settings ⚙️ -> Database -> Connection Info)
USER = "postgres"
PASSWORD = "TU_CONTRASEÑA_DE_SUPABASE_AQUÍ"  # <-- Coloca tu contraseña real aquí
HOST = "TU_HOST_DE_SUPABASE_AQUÍ"            # <-- Coloca tu Host largo aquí (ej: aws-0-...)
PORT = "5432"
DBNAME = "postgres"

# Creamos el motor de conexión a la base de datos
engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}")

print("🔌 Conectando a Supabase y descargando datos...")
# Leemos la tabla que acabas de subir
df = pd.read_sql("SELECT * FROM datos_origen", engine)
print(f"✅ ¡Datos cargados! Total de registros: {len(df)}")

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO
# En tu dataset, 'quality' es la columna real. 
# Crearemos una regla de negocio: Si la calidad es 6 o más, es un vino "Bueno" (1). Si es menos, es "Regular" (0).
df['clase_real'] = (df['quality'] >= 6).astype(int)

# Seleccionamos las columnas numéricas químicas que usará KNN para aprender
# Usaremos el nivel de alcohol, la acidez (volatile acidity) y los sulfatos como ejemplos clave
features = ['volatile acidity', 'alcohol', 'sulphates', 'citric acid']
X = df[features]
y = df['clase_real']

# El algoritmo KNN se basa en distancias, por lo que es OBLIGATORIO escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. ENTRENAMIENTO DEL CLASIFICADOR KNN
print("🧠 Entrenando el modelo KNN (K=5)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# 4. GENERAR PREDICCIONES
df['prediccion_knn'] = knn.predict(X_scaled)

# Evaluamos rápidamente qué tan bueno fue el modelo
precision = accuracy_score(df['clase_real'], df['prediccion_knn'])
print(f"🎯 Precisión del modelo en el entrenamiento: {precision * 100:.2f}%")

# Guardamos si el algoritmo acertó o falló en cada registro para graficarlo en Power BI
df['es_correcto'] = df['clase_real'] == df['prediccion_knn']

# 5. SUBIR LOS RESULTADOS A UNA NUEVA TABLA EN SUPABASE
print("📤 Subiendo la tabla clasificada a Supabase...")
# Esto creará una tabla nueva llamada 'resultados_knn' de forma automática
df.to_sql('resultados_knn', engine, if_exists='replace', index=False)
print("🎉 ¡Proceso terminado con éxito! Tabla 'resultados_knn' lista en la nube.")