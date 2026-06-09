# Arquitetura — World Cup Agent

## Fase 1: Grafo Base

### Fluxo de Execução

```
Usuário digita pergunta
        │
        ▼
┌─────────────────────┐
│   input_processor   │  Normaliza texto, cria HumanMessage,
│                     │  incrementa iteração, adiciona metadata
└──────────┬──────────┘
           │ (sempre)
           ▼
┌─────────────────────┐
│  query_classifier   │  Analisa palavras-chave, categoriza
│                     │  (campeao/artilheiro/historia/etc)
└──────────┬──────────┘
           │ (condicional)
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ response│  │ error_handler│
│generator│  │              │
└────┬────┘  └──────┬───────┘
     │ (condicional)│ (sempre)
     └──────┬───────┘
            ▼
          [END]
            │
            ▼
     Resposta ao usuário
```

### Estado do Grafo (GraphState)

O estado flui pelos nós sendo enriquecido progressivamente:

```
Entrada:
  user_input = "Quem ganhou 1970?"
  agent_status = "pending"
  messages = []

Após input_processor:
  processed_query = "quem ganhou 1970?"
  agent_status = "processing"
  messages = [HumanMessage("Quem ganhou 1970?")]
  iteration_count = 1

Após query_classifier:
  metadata = {"query_category": "campeao"}
  agent_status = "processing"

Após response_generator:
  final_response = "Brasil é o maior campeão..."
  agent_status = "success"
  messages = [HumanMessage(...), AIMessage("Brasil é...")]
```

### Decisões de Design

**Por que TypedDict para o estado?**
LangGraph requer TypedDict para que o runtime possa fazer merge parcial
de atualizações. Quando um nó retorna `{"processed_query": "..."}`,
o LangGraph sabe que só deve atualizar esse campo, mantendo os outros.

**Por que Annotated[list, operator.add] para messages?**
Sem o Annotated, o último valor ganha. Com `operator.add`,
as listas são concatenadas. Isso é fundamental para o histórico
de conversa crescer corretamente.

**Por que Singleton no get_graph()?**
Construir o grafo tem um custo (validação, compilação). Fazemos
isso uma vez na inicialização e reutilizamos a instância compilada
em todas as requisições subsequentes.
