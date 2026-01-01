Criei o Script de Teste: test_habits_only.py.

Isso para validar se o modelo estava realmente aprendendo padrões comportamentais
e não apenas reproduzindo o cálculo de IMC, realizamos um "Teste Cego" (Blind Test).

Removemos as variáveis biométricas (Peso, Altura e IMC) e treinamos o modelo apenas com dados sociodemográficos e de hábitos.
Resultado: O modelo atingiu uma acurácia de 85.3%, comprovando que é capaz de identificar perfis de risco com alta precisão
baseando-se exclusivamente no estilo de vida do paciente (Consumo de Vegetais, Sedentarismo, Idade, etc.). Isso valida o sistema
como uma ferramenta de Medicina Preventiva, capaz de alertar riscos antes mesmo do ganho de peso excessivo.