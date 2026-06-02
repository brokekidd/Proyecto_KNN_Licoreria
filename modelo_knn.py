import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. CONEXIÓN DIRECTA A SUPABASE (Usando tu DIRECT_URL)
# REEMPLAZA únicamente donde dice TU_CONTRASEÑA_REAL por la tuya.
DATABASE_URL = "postgresql://postgres.fyafpnztpihuthlhauif:HatoRey4$$$@aws-1-us-east-2.pooler.supabase.com:5432/postgres"

# Creamos el motor de conexión
engine = create_engine(DATABASE_URL)

try:
    print("🔌 Conectando a Supabase y descargando datos...")
    # Leemos la tabla que subiste mediante el CSV
    df = pd.read_sql("SELECT * FROM datos_origen", engine)
    print(f"✅ ¡Datos cargados con éxito! Total de registros: {len(df)} vinos.")

    # 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO KNN
    # Regla de negocio para la licorería: Si la calidad ('quality') es 6 o más, es vino Premium (1), si no, es Regular (0).
    df['clase_real'] = (df['quality'] >= 6).astype(int)

    # Seleccionamos las variables químicas del vino para que KNN calcule las distancias
    features = ['volatile acidity', 'alcohol', 'sulphates', 'citric acid']
    X = df[features]
    y = df['clase_real']

    # KNN se basa en distancias espaciales; es OBLIGATORIO normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. ENTRENAMIENTO DEL CLASIFICADOR KNN
    print("🧠 Entrenando el modelo KNN (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)

    # 4. GENERAR PREDICCIONES
    df['prediccion_knn'] = knn.predict(X_scaled)

    # Evaluamos la precisión del algoritmo
    precision = accuracy_score(df['clase_real'], df['prediccion_knn'])
    print(f"🎯 ¡Modelo entrenado! Precisión (Accuracy): {precision * 100:.2f}%")

    # Guardamos si el algoritmo acertó o falló para mapearlo en Power BI
    df['es_correcto'] = df['clase_real'] == df['prediccion_knn']

    # 5. SUBIR LOS RESULTADOS A SUPABASE
    print("📤 Subiendo la nueva tabla clasificada 'resultados_knn' a Supabase...")
    df.to_sql('resultados_knn', engine, if_exists='replace', index=False)
    print("🎉 ¡PROCESO COMPLETADO! Revisa tu panel de Supabase, la tabla ya debe estar ahí.")

except Exception as e:
    print(f"❌ Ocurrió un error en la conexión: {e}")

    # Guarda una copia exacta de los resultados en un archivo CSV en tu carpeta
df.to_csv('resultados_knn_licoreria.csv', index=False, encoding='utf-8')
print("💾 ¡Archivo CSV local generado con éxito!")