reasoning_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
ordinary_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant provides the user with the answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>."

reasoning_user = """
Code Search Query:
"{target_query}"

Candidate Code Snippets:
{candidate_blocks}
Task:

As a professional software developer, please rank the N={num} candidate code segments based on their relevance to the provided code query. The query is a code snippet of which the next statement needs to be completed, and the segments are contextual pieces that might help in completing the code.
A segment is stated by a candidate index. 

Requirements:
1. Score each candidate segment based on its relevance to the query (1-10 scale, with 1 being the lowest and 10 being the highest);
2. When ranking, consider how well the context provided by each segment aids in completing the code, including keyword relevance, logical flow, and completeness of the information;
3. Make sure the result contains all the segment index and its ranking score and ensure that no index is repeated multiple times;
4. Must list all candidate snippets, with no additions or omissions.

Strict Output Format:

<think>  
Relevance reasoning ...
</think>  
<answer>[Rank1](Score1) > [Rank2](Score2) > ... > [RankN](ScoreN)</answer>

Example:
If there are 5 candidates and you assign them scores of 6, 1, 3, 5, and 9 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[5](9) [1](6) [4](5) [3](3) [2](1)</answer>.
If there are 4 candidates and you assign them scores of 9, 4, 7, and 10 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[4](10) [1](9) [3](7) [2](4)</answer>.
If there are 8 candidates and you assign them scores of 7, 8, 4, 1, 10, 5, 3 and 10 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[5](10) [8](10) [2](8) [1](7) [6](5) [3](4) [7](3) [4](1)</answer>.
"""

ordinary_user = """
Code Search Query:
"{target_query}"

Candidate Code Snippets:
{candidate_blocks}
Task:

As a professional software developer, please rank the N={num} candidate code segments based on their relevance to the provided code query. The query is a code snippet of which the next statement needs to be completed, and the segments are contextual pieces that might help in completing the code.
A segment is stated by a candidate index. 

Requirements:
1. Score each candidate segment based on its relevance to the query (1-10 scale, with 1 being the lowest and 10 being the highest);
2. When ranking, consider how well the context provided by each segment aids in completing the code, including keyword relevance, logical flow, and completeness of the information;
3. Make sure the result contains all the segment index and its ranking score and ensure that no index is repeated multiple times;
4. Must list all candidate snippets, with no additions or omissions.

Strict Output Format:

<answer>[Rank1](Score1) [Rank2](Score2)  ...  [RankN](ScoreN)</answer>

Examples:
If there are 5 candidates and you assign them scores of 6, 1, 3, 5, and 9 respectively, your answer should look like this: <answer>[5](9) [1](6) [4](5) [3](3) [2](1)</answer>.
If there are 4 candidates and you assign them scores of 9, 4, 7, and 10 respectively, your answer should look like this: <answer>[4](10) [1](9) [3](7) [2](4)</answer>.
If there are 8 candidates and you assign them scores of 7, 8, 4, 1, 10, 5, 3 and 10 respectively, your answer should look like this: <answer>[5](10) [8](10) [2](8) [1](7) [6](5) [3](4) [7](3) [4](1)</answer>.
"""

reasoning_user_set_wise = """
Given the query below, which of the following documents is most relevant? 
query:
{query}
documents:
{documents}
After completing the reasoning process, please provide only the label of the most relevant document to the query, enclosed in square brackets, within the answer tags. \
For example, if the document C is the most relevant, the answer should be: <think> reasoning process here </think> <answer>[C]</answer>.
"""

user_set_wise = """
Given the query below, which of the following documents is most relevant? 
query:
{query}
documents:
{documents}
Please provide only the label of the most relevant document to the query, enclosed in square brackets, within the answer tags. For example, if the document C is the most relevant, the answer should be: <answer>[C]</answer>.
"""