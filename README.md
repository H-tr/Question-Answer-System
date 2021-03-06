# Question-Answer-System
## Introduction
This is a question-answer system project. This project is divided into few parts:
- Simple GPT and BERT examples to compare and choose which one to use
- Question-answer system based on GPT
- Question-answer system based on BERT
- Android app developed based on one question-answer system
## Requirement
- OS: Ubuntu
- Python >= 3.9
- openai
- pytorch >= 1.11
- transformers
## GPT example
This GPT example could transform natural language into sql instructions.
The result of this example is:
```
The output for "Display the lowest salary from the Worker table" is:
output: Select min(Salary) from Worker where Salary not in (Select min(Salary) from Worker);


The output for "Tell me the count of employees working in the department HR." is:
output: Select count(*) from Worker where DEPARTMENT = 'HR';


The output for "Get salary details of the Workers whose AGE lies between 25 and 35" is:
output: Select Salary from Worker where AGE between 25 and 35;
```
### Usage
To run the example, enter `GPTtry` and run
```
python GPT3.py
```
## BERT example
The BERT example is a question-answer system with only one training data. 
It is implemented with the help of [the guide](https://towardsdatascience.com/question-answering-with-pretrained-transformers-using-pytorch-c3e7a44b4012) Provides the description of machine learning:
> Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

Set the question as

> What are machine learning models based on?

We can get this output:
```
Answer: sample data,
Score: 0.8846667408943176
```
### Usage
To run the example, enter `BERTtry` and run
```
python BERT.py
```

## GPT based question-answer system
The GPT based question-answer system is implemented with the help of [this guide](https://www.twilio.com/blog/ultimate-guide-openai-gpt-3-language-model).
It could answer some simple questions. For example:
```
You: What is python?
GPT-3: Python is a programming language.
You: Do you know how to code?
GPT-3: Yes, I know how to code.
You: How to code?
GPT-3: In Python, the most basic way to code is to use the print statement.
You: Which language should I learn first?
GPT-3: I think it depends on what you want to do with coding. For example, if you want to make websites, you might want to learn HTML and CSS first.
```
### Usage
To run the Question-Answer system, enter `QAsystem-GPT` and run
```
python chat.py
```
## BERT based question-answer system
The BERT based question-answer system is implemented with the help of [this guide](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626).
We use an already available fine-tund BERT model from the Hugging Face Transformers library to 
answer questions based on the stories from the CoQA dataset.
### Usage
To run the Question-Answer system, enter `QAsystem-BERT` and run
```
python QAsystemBERT.py
```
It will first display the number of question and answers pairs to train this model:
```
Number of question and answers: 108647
```
Then the model will ask you to enter a paragraph of text. You can use the following example:
```
The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula.   The Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail.   In March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online.   The Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items.   Scholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican.   The Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.
```
And then you can ask questions with the guide. There is a little demo:
```
Please enter your question: 
When was the Vat formally opened?

Predicted answer:
1475

Do you want to ask another question based on this text (Y/N)?Y

Please enter your question: 
What is the library for?

Predicted answer:
Research library for history , law , philosophy , science and theology

Do you want to ask another question based on this text (Y/N)?Y

Please enter your question: 
for what subjects?

Predicted answer:
History , law , philosophy , science and theology

Do you want to ask another question based on this text (Y/N)?N

Bye!
```
## Knowledge based Question and Answer system
The knowledge based QA system is based on BERT. It find data from Wikipedia and get the exact 
answer. The more detailed information is (here)[./KnowledgeBased/].
### Open domain QA system
#### Usage
To run this program, enter `KnowledgeBased/OpenDomain` and run
```
python chatbot.py
```
You can ask any question. For example:
> Why is the sky blue?

The output should be like this:
```
Please enter your question: 
When was Barack Obama born?
Top wiki result: <WikipediaPage 'Family of Barack Obama'>
Answer: january 17, 1964 / 

Do you want to ask another question (Y/N)?Y

Please enter your question: 
Why is the sky blue?
Top wiki result: <WikipediaPage 'Diffuse sky radiation'>
Answer: wavelengths are shorter / rayleigh scattering / 

Do you want to ask another question (Y/N)?Y

Please enter your question: 
How many sides does a pentagon have?
Top wiki result: <WikipediaPage 'The Pentagon'>
Answer: five / 

Do you want to ask another question (Y/N)?N

Bye!
```
If you see this warning:
```
Token indices sequence length is longer than the specified maximum sequence length for this model (9267 > 512). Running this sequence through the model will result in indexing errors
```
Just ignore it.
In the above example, the answer for the first question is not correct, This is because wikipedia's
search engine returns uncorrect document. The document reader system works very well.

### Closed domain QA system
#### Using cosine similarity to choose the file
To run this program, enter `KnowledgeBased/CloseDomain` and run
```
python QAsystem.py
```
You can ask question like
> What is the size of output pictures in AlexNet?

The output should be:
```
Please enter your question: 
What is the size of output picture in AlexNet?
[nltk_data] Downloading package stopwords to /home/run/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Computing similarity of Transformer.pdf
100%|████████████████████████████████████████████████████████████████| 1987/1987 [02:55<00:00, 11.32it/s]
The similarity of Transformer.pdf is 0.15155404806137085
Computing similarity of AlexNet.pdf
100%|████████████████████████████████████████████████████████████████| 1785/1785 [02:43<00:00, 10.91it/s]
The similarity of AlexNet.pdf is 0.16987115144729614
Computing similarity of RestNet.pdf
100%|████████████████████████████████████████████████████████████████| 2698/2698 [03:41<00:00, 12.20it/s]
The similarity of RestNet.pdf is 0.069475457072258
Computing similarity of GPT3.pdf
100%|████████████████████████████████████████████████████████████████| 9009/9009 [12:29<00:00, 12.02it/s]
The similarity of GPT3.pdf is 0.06203676387667656
Computing similarity of GAN.pdf
100%|████████████████████████████████████████████████████████████████| 1607/1607 [02:18<00:00, 11.61it/s]
The similarity of GAN.pdf is 0.06314083188772202
Computing similarity of MAE.pdf
100%|████████████████████████████████████████████████████████████████| 3039/3039 [03:52<00:00, 13.05it/s]
The similarity of MAE.pdf is 0.12731876969337463
Computing similarity of ViT.pdf
100%|████████████████████████████████████████████████████████████████| 3149/3149 [03:59<00:00, 13.14it/s]
The similarity of ViT.pdf is 0.1577579826116562

BERT: 1.2 million
```
The answer is correct but it takes too much time.
#### Using simhash to choose the file
The output is
```
Please enter your question: 
What does transformer based on?
The distance to Transformer.pdf is 28
The distance to AlexNet.pdf is 28
The distance to RestNet.pdf is 28
The distance to GPT3.pdf is 27
The distance to GAN.pdf is 26
The distance to MAE.pdf is 26
The distance to ViT.pdf is 32
No model was supplied, defaulted to distilbert-base-cased-distilled-squad (https://huggingface.co/distilbert-base-cased-distilled-squad)
BERT: Markov chains
```
It is much faster than cosine similarity. But it might not that accuracy and might have many documents has the same distance value.
#### Combine simHash and cosine similarity together
The output is:
```
Please enter your question:
How large is the patch of resulting image in AlexNet?
The distance to Transformer.pdf is 26
The distance to AlexNet.pdf is 24
The distance to RestNet.pdf is 24
The distance to GPT3.pdf is 29
The distance to GAN.pdf is 28
The distance to MAE.pdf is 24
The distance to ViT.pdf is 26
AlexNet.pdf is selected.
RestNet.pdf is selected.
MAE.pdf is selected.
[nltk_data] Downloading package stopwords to /home/run/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
100%|███████████████████████████████████████████████████████████████████████████████| 1784/1784 [01:13<00:00, 24.25it/s]
Selected
The similarity is 0.18773122131824493
100%|███████████████████████████████████████████████████████████████████████████████| 2698/2698 [01:38<00:00, 27.28it/s]
The similarity is 0.08289150148630142
100%|███████████████████████████████████████████████████████████████████████████████| 3038/3038 [01:44<00:00, 29.02it/s]
The similarity is 0.17737512290477753
BERT: 224 × 224 × 3
Do you want to ask another question? (Y/n) y

Please enter your question:
What is the optimal masking ratio in MAE?
The distance to Transformer.pdf is 33
The distance to AlexNet.pdf is 31
The distance to RestNet.pdf is 29
The distance to GPT3.pdf is 32
The distance to GAN.pdf is 29
The distance to MAE.pdf is 29
The distance to ViT.pdf is 33
RestNet.pdf is selected.
GAN.pdf is selected.
MAE.pdf is selected.
[nltk_data] Downloading package stopwords to /home/run/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
100%|███████████████████████████████████████████████████████████████████████████████| 2698/2698 [01:43<00:00, 26.12it/s]
Selected
The similarity is 0.01940499059855938
100%|███████████████████████████████████████████████████████████████████████████████| 1606/1606 [01:03<00:00, 25.28it/s]
Selected
The similarity is 0.06007204204797745
100%|███████████████████████████████████████████████████████████████████████████████| 3038/3038 [01:47<00:00, 28.18it/s]
Selected
The similarity is 0.14355123043060303
BERT: 75%
Do you want to ask another question? (Y/n) n
Bye!
```
## Android app

The detailed progress information could be find in `plan`.
