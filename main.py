from src.load_qa_dataset import load_qa_dataset, sample_qa

qa = load_qa_dataset()

sample = sample_qa(qa, n=20)

for category, df in sample.items():
    print(f"{category} sample: {len(df)} rows")
    for _, row in df.head().iterrows():
        print (f'En el curso {row["Curso"]} de la {row["Maestría"]}, {row["Pregunta"]} -> {row["Respuesta"]}\n')