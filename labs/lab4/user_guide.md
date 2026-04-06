# User Guide — Lab 4: LangChain Applications

## 1. Setup

Install the required Python packages:

```bash
pip install langchain langchain-groq langchain-community langchain-huggingface \
            faiss-cpu sentence-transformers wikipedia arxiv python-dotenv
```

## 2. API Keys

Create a `.env` file in the `labs/lab4/` directory with your keys:

```
GROQ_API_KEY=your_groq_api_key_here
```

The code loads these automatically via `python-dotenv`.

## 3. Running the Applications

### App 1 — Enterprise Customer Support

```bash
cd labs/lab4
python app1_customer_support.py
```

The chatbot will start an interactive session. Type your question and press Enter.

### App 2 — Academic Research Assistant

```bash
cd labs/lab4
python app2_research_assistant.py
```

You will be prompted for a topic and a report format (`brief`, `detailed`, or `academic`).

---

## 4. Example Usage Scenarios

### App 1 — Customer Support Examples

**Scenario 1: Asking about a product**
```
 You: What are the camera specs of the TechNova X200?

 Support:
--------------------------------------------------
 Category   : Knowledge Base Query
--------------------------------------------------
 Answer     :
The TechNova X200 features a quad-camera system:
- Main: 200MP wide lens with OIS
- Ultra-wide: 12MP (120° FOV)
- Telephoto: 50MP with 100x Space Zoom
- Front: 32MP with autofocus
It also supports 8K video recording at 30fps.
--------------------------------------------------
 Sources    : product_a.txt
--------------------------------------------------
 Suggested Actions : Ask a follow-up question or type 'quit' to exit.
--------------------------------------------------
```

**Scenario 2: Tracking an order**
```
 You: Can you track my order 12345?

 Support:
--------------------------------------------------
 Category   : Order Tracking
--------------------------------------------------
 Answer     :
Order #12345: Shipped — expected delivery on March 12, 2026.
--------------------------------------------------
 Suggested Actions : If the status seems wrong, contact support@technova.com.
--------------------------------------------------
```

**Scenario 3: Checking stock availability**
```
 You: Is the wireless mouse in stock?

 Support:
--------------------------------------------------
 Category   : Stock Availability
--------------------------------------------------
 Answer     :
'wireless mouse m100': Out of Stock — expected restock on March 20, 2026.
--------------------------------------------------
 Suggested Actions : Visit technova.com to place an order.
--------------------------------------------------
```

**Scenario 4: Policy question**
```
 You: What is your return policy?

🤖 Support:
Customers may return any product within 30 days of purchase for a full refund.
Products must be in original packaging with no signs of damage.
(Sources: company_policy.txt)
```

---

### App 2 — Research Assistant Examples

**Scenario 1: Brief report**
```
 Enter research topic: AI in healthcare
 Report format (brief/detailed/academic) [detailed]: brief

=======================================================
## Summary
Artificial Intelligence is transforming healthcare through ...

## Key Takeaways
- AI-driven diagnostics improve accuracy by up to 20%
- Natural language processing enables automated medical records
- Challenges include data privacy and regulatory compliance

## References
- Wikipedia: https://en.wikipedia.org/wiki/AI_in_healthcare
[1] Deep Learning in Medical Imaging. Smith et al. Published: 2024-06-15. URL: https://arxiv.org/abs/2406.12345
=======================================================
```

**Scenario 2: Academic report**
```
 Enter research topic: transformer neural networks
 Report format (brief/detailed/academic) [detailed]: academic

=======================================================
## Abstract
This report examines the transformer architecture ...

## 1. Introduction
...

## 2. Methodology
Information was gathered from Wikipedia for general context
and ArXiv for peer-reviewed research papers ...

## 3. Findings
...

## 4. Analysis & Discussion
...

## 5. Conclusion
...

## References
- Wikipedia: https://en.wikipedia.org/wiki/Transformer_neural_networks
[1] Attention Is All You Need. Vaswani et al. Published: 2017-06-12. URL: https://arxiv.org/abs/1706.03762
[2] BERT: Pre-training of Deep Bidirectional Transformers. Devlin et al. ...
=======================================================
```

---

## 5. Notes

*Note: Models used are Llama 3.1 8B via the Groq API for fast inference.*
- **Knowledge Base**: App 1 reads all `.txt` files from the `./knowledge` folder. You can add more files to expand the support system's expertise.
- **Report Formats**: App 2 supports `brief` (quick summary), `detailed` (full analysis), and `academic` (formal paper structure).
