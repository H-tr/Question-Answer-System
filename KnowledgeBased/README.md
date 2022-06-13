# Knowledge based Question and Answer System
There are two kinds of Question and Answer System: open domain and closed domain. Open domain QA system is to 
retrieve informations from a non-bounded domain of text, and closed domain QA system is to retrieve from a specific domain of text.

The knowledge given to the QA system is in unstructured formats. The QA system needs to use 
some deep learning models like BERT or GPT3 to parse those data into structured formats like SQL.

This knowledge based QA system section, we will implement two parts:
- Open domain trained from Wikipedia
- Close domain trained from specific documents
## Information Retreval-Based Systems
Information retrieval-based question answering (IR QA) systems find and extract a text segment from a large
collection of documents. It first idenfity the most relevant documents in the collection, and then extract the 
answer from the contents of those documents.

Different from search engines, the QA system not only provide the most likely documents for the question asked, 
but also process the language in the top few documents to get the more specific answer. In general, the workflow 
could be described as two components:
- documents retriever
- documents reader

## Reference
[Introduction to QA System](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html)
[Building a QA System with BERT on Wikipedia](https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html)

