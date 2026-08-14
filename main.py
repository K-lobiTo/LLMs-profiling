from utils.load_qa_dataset import load_qa_dataset, sample_qa

# print(len(corpus))

# docIdx = 5
# print(corpus[list(corpus.keys())[docIdx]])


qa = load_qa_dataset()
# for category, df in qa.items():
#     print(f"{category}: {len(df)} questions")

# for _, row in qa["binary"].head(2).iterrows():
#     print(row["Pregunta"], "->", row["Respuesta"])

sample = sample_qa(qa, n=20)              # 20 random rows per category, no fixed seed (different each run)
# sample = sample_qa(qa, n=20, seed=42)     # reproducible sample, same 20 rows every time

sample["binary"]        # 20 random yes/no questions
sample["short_answer"]  # 20 random short-answer questions
sample["open_ended"]    # 20 random open-ended questions


for category, df in sample.items():
    print(f"{category} sample: {len(df)} rows")
    for _, row in df.head().iterrows():
        print (f'En el curso {row["Curso"]} de la {row["Maestría"]}, {row["Pregunta"]} -> {row["Respuesta"]}\n')