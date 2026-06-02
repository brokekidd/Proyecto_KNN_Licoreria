import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Configuracion de la conexion al servidor de base de datos
DATABASE_URL = "postgresql://postgres.fyafpnztpihuthlhauif:HatoRey4$$$@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL)

try:
    print("Conectando al servidor y descargando datos de origen...")
    df = pd.read_sql("SELECT * FROM datos_origen", engine)
    print(f"Lectura exitosa. Total de registros: {len(df)} muestras.")

    # 2. Preprocesamiento y definicion de la variable objetivo
    # Criterio de clasificacion: Calidad >= 6 se define como Premium (1), de lo contrario Regular (0)
    df['clase_real'] = (df['quality'] >= 6).astype(int)

    # Seleccion de variables quimicas como descriptores
    features = ['volatile acidity', 'alcohol', 'sulphates', 'citric acid']
    X = df[features]
    y = df['clase_real']

    # Normalizacion de caracteristicas (Escalamiento obligatorio para KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Ajuste del modelo Clasificador KNN
    print("Entrenando el modelo K-Nearest Neighbors (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)

    # 4. Evaluacion del modelo y generacion de predicciones
    df['prediccion_knn'] = knn.predict(X_scaled)
    precision = accuracy_score(df['clase_real'], df['prediccion_knn'])
    print(f"Modelo entrenado exitosamente. Precision (Accuracy): {precision * 100:.2f}%")

    # Mapeo de aciertos para analisis en la capa de presentacion
    df['es_correcto'] = df['clase_real'] == df['prediccion_knn']

    # 5. Exportacion y carga de resultados hacia los destinos correspondientes
    print("Exportando la tabla 'resultados_knn' hacia Supabase...")
    df.to_sql('resultados_knn', engine, if_exists='replace', index=False)
    
    print("Generando respaldo local en formato CSV...")
    df.to_csv('resultados_knn_licoreria.csv', index=False, encoding='utf-8')
    
    print("Proceso de ejecucion, clasificacion y exportacion completado.")

except Exception as e:
    print(f"Ocurrio un error en el flujo de ejecucion: {e}")