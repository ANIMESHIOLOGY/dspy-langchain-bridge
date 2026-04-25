# Optimizer Bridge

Optimize a LangChain prompt using DSPy's `BootstrapFewShot`, `MIPROv2`, or `COPRO`.

## Example

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dspy_lc_bridge import LangChainLM, optimize_langchain_prompt

dspy.settings.configure(lm=LangChainLM(ChatOpenAI(model="gpt-4o")))

# 1. Define your prompt
prompt = ChatPromptTemplate.from_template("Answer the question: {question}")

# 2. Define a metric
def exact_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# 3. Build your trainset
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
]

# 4. Optimize
optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=3)
optimized_prompt = optimize_langchain_prompt(
    prompt=prompt,
    optimizer=optimizer,
    trainset=trainset,
)

# 5. Use the optimized prompt in your chain
from langchain_openai import ChatOpenAI
chain = optimized_prompt | ChatOpenAI(model="gpt-4o")
result = chain.invoke({"question": "What is 3+3?"})
```

## How It Works

1. The LangChain `PromptTemplate` is converted to a temporary DSPy `Signature`.
2. A minimal `dspy.Predict` module wrapping that Signature is created.
3. The DSPy optimizer runs on the module using your trainset and metric.
4. After optimization, few-shot demonstrations are extracted from the compiled module.
5. A new `ChatPromptTemplate` is returned with those demos injected as human/assistant message pairs and any new instructions added to the system message.
