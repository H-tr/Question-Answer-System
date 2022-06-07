import json
import openai

with open('GPT_SECRET_KEY.json') as f:
    data = json.load(f)

openai.api_key = data['API_KEY']
from gpt import GPT
from gpt import Example

gpt = GPT(engine="davinci",
          temperature=0.5,
          max_tokens=100)

gpt.add_example(Example("Fetch unique value of DEPARTMENT from Worker table.",
                        "Select distinct DEPARTMENT from Worker;"))

gpt.add_example(Example("Print the first three characters of FIRST_NAME from Worker table.",
                        "Select substring(FIRST_NAME, 1, 3) from Worker;"))

gpt.add_example(Example("Find the position of the alphabet ('a') in the first name column 'Amitabh' from Worker table.",
                        "Select INSTR(FIRST_NAME, BINARY'a) from Worker where FIRST_NAME = 'Amitabh';"))

gpt.add_example(Example("Print the FIRST_NAME from Worker table after replacing 'a' with 'A'.",
                        "Select CONCAT(FIRST_NAME, ' ', LAST_NAME) AS 'COMPLETE_NAME' from Worker;"))

gpt.add_example(Example("Display the second highest salary from the Worker table.",
                        "Select max(Salary) from Worker where Salary not in (Select max(Salary) from Worker);"))

gpt.add_example(Example("Display the highest salary from the Worker table.",
                        "Select max(Salary) from Worker where Salary not in (Select max(Salary) from Worker);"))

gpt.add_example(Example("Fetch the count of employees working in the department Admin.",
                        "SELECT COUNT(*) FROM worker WHERE DEPARTMENT = 'Admin';"))

gpt.add_example(Example("Get all details of the Workers whose SALARY lies between 100000 and 500000.",
                        "Select * from Worker where SALARY between 100000 and 500000;"))

gpt.add_example(Example("Get Salary details of the Workers.",
                        "Select Salary from Worker;"))
# Example 1
prompt = "Display the lowest salary from the Worker table."
output = gpt.submit_request(prompt)
print("The output for \"Display the lowest salary from the Worker table\" is:")
print(output.choices[0].text)

# Example 2
prompt = "Tell me the count of employees working in the department HR."
output = gpt.submit_request(prompt)
print("The output for \"Tell me the count of employees working in the department HR.\" is:")
print(output.choices[0].text)

# Example 3
prompt = "Get salary details of the Workers whose AGE lies between 25 and 35"
print("The output for \"Get salary details of the Workers whose AGE lies between 25 and 35\" is:")
print(gpt.get_top_reply(prompt))
