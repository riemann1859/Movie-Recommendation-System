**Project: Design a Recommendation System**

Recommendation Systems are an increasingly popular application of Machine Learning in many industries. If you've used services like 
Amazon, Netflix or Pandora, you might notice their personalized recommendations suggesting items for you to buy, movies to watch, 
or songs to listen to. An example of how influential a well-designed recommendation system can be is what happened to Joe Simpson's 
Touching the Void, a book about mountain-climbing. The book was not a bestseller when it was published originally in 1988, but 
surged in popularity after another mountain-climbing book, Into Thin Air by Jon Krakauer, topped the charts in 1997. 
This happened because Amazon's recommendation system noticed that a few people who bought and enjoyed Into Thin Air also bought 
and enjoyed Touching the Void. Without good online recommendation systems to suggest the book to readers, 
Touching the Void may have been quickly forgotten. Instead, it became very popular in its own right and in some respects 
exceeded the popularity of Into Thin Air.

Similarly, in this project, we will recommend movies to users. We will use the MovieLens dataset 
(https://www.udacity.com/api/nodes/1021838736/supplemental_media/ml2-projectzip/download  or https://grouplens.org/datasets/movielens/)
to create a recommendation engine like Netflix's. We will 
then use the Clustering Algorithms  to refine our engine to make better and better recommendations.


We will be using the Pearson Correlation Similarity Measure as the basis for designing our recommendation engine, 
and then refine predictions using clustering algorithms.

In Part 1, you will use your programming skills to produce a utility matrix predicting the ratings that 
users will give to items. You can verify your solution by submitting predicted user ratings in this matrix. 
To measure the difference between your predicted values and a set of given ratings, 
you will compute the Mean Squared Error here (see Part 1 Instructions). 
In Part 2, you will use clustering algorithms to refine your utility matrix. You will have the chance to test your predictions, and see if your score passes the threshold here (see Part 2 Instructions).
In Part 3, you will answer questions to reflect on what you have done in Part 2 to improve your predictions.
Assessment
Parts of this project will be automatically verified through the Udacity autograder, including utility matrix values and predictions. For students enrolled in the full course experience, your coach will assess your performance on the project using the rubric linked below, in addition to automatic verification.

Rubric
You can view the rubric here. Be sure to review it to make sure your project is complete before submitting it.

Part 1: Collaborative Filtering
Recommendation engines generally use two different techniques to make predictions:

Content-Based Systems examine properties of the items recommended. For example, if a Netflix user has watched many sci-fi movies, then recommend a movie from the â€œsci-fiâ€ genre.
Collaborative Filtering Systems recommend items based on similarity measures between users and/or items. The items they recommend to a user are drawn from those preferred by similar users.  For example, if an Amazon shopper places a toy robot in their cart, then a robot-themed Lego set is recommended because other shoppers who liked the toy robot also liked the Lego set.
We will use Collaborative Filtering to build our recommendation engine.

We will start by mathematically formulating a recommendation engine framework.  In a recommendation-system application, there are two classes of entities, users and items. Users have preferences for certain items and these preferences must be discovered from the data. The data is represented as a utility matrix, a value that represents the rating given by that user for that item and is given for each user-item pair. The ratings can take a value between 1 to 5, representing the number of stars that the user gave as a rating for that item.

We will assume that the matrix is sparse, meaning that most of the entries are unknown.

Here is an example of a utility matrix, depicting four users: Ann, Bob, Carl and Doug. The available movie names are HP1, HP2 and HP3 for the Harry Potter movies, and SW1, SW2 and SW3 for Star Wars episodes 1, 2 and 3.

Table 1. An example Utility Matrix for Movie Recommendation (you can fill this out here) proj-part1-q1_utility-matrix.png

The goal of the recommendation engine is to predict the blanks in a utility matrix. For example, how will Ann rate SW2?

In order to make these predictions, we must first measure similarity of users or items from the rows and columns of the Utility Matrix. The data is too small to draw any reliable conclusions. Examine the preferences of Ann and Carl. They both have rated the same movie but have diametrically opposite opinions on it. We would expect that a good distance measure would set them far apart. We can use different similarity measures for this task, but in this project we will use the Pearson Correlation Similarity Measure.

Let $r_{x,i}$ denote the rating given by user $x$ to item $i$. If $I$ is the set of all items that two users $x$ and $y$ have both rated, then the Pearson Correlation Similarity Measure between the two users is given by:

pcs.png

where $\overline{r_x}$ denotes the average rating given by user $x$ to all items. To calculate $\overline{r_x}$ we only consider items that were rated by the user. If there are no common items that both users have rated (i.e. $I$ is empty), we set pcs(x, y) to $0$.

We still need to take an additional step to recommend items to users. One way of predicting the value of the utility matrix entry (estimated rating) of a given user u for item i, is to find the n number of users most similar to u and average their ratings for item i, counting only those among the n similar users who have rated i. Note that you must be careful not to compare the user to themselves when finding these top n similar users.

When making predictions, it is generally better to normalize the matrix first. That is, for each of the n most similar users, subtract their average rating for all items from the rating of the item of interest i. This tells us whether a user rates the movie above the average movie they have rated or below that average. Take the average of these differences for those users who have rated i, then add this average difference to the average rating that u gives for all items. This normalization adjusts the estimate in the case that u tends to give very high or very low ratings; it calibrates the estimate more to the individual's tendencies. A useful side-effect of this approach is that when none of the n most similar users have rated item i, then the best guess reduces to the average rating that u has given across all items.

Part 1 Instructions:

Fill the Utility Matrix in Table 1.

a) Download the code template (ml2_project.zip under Downloadables section here), and edit part1.py.

b) Complete the function pcs(x, y). It finds the Pearson Correlation Similarity Measure between two users.

c) Complete the function guess(u, i, top_n). It finds the best guess of a rating for user u and item i, if we look at the top_n users similar to user u.

d) Run the script (python part1.py) to guess all the blanks. You can verify your answers by submitting them here.

Note: This project uses a 1-based indexing scheme, i.e. the first items in the users and items lists have an ID of 1 (this is how the MovieLens dataset used in Part 2 is encoded, and we want to follow the same pattern in Part 1).

The script part1.py also includes a comparison with the following test set. Report the Mean Squared Error displayed and verify your answer here.

a) Ann rates SW2 with 2 stars

b) Carl rates HP1 with 2 stars

c) Carl rates HP2 with 2 stars

d) Doug rates SW1 with 4 stars

e) Doug rates SW2 with 3 stars

Can we improve the accuracy of our predictions if we work from similar items instead of similar users? What are the pros/cons of this approach? Explain in 4-5 sentences.

(Optional) Cluster similar items using Pearson Correlation Similarity Measure and report the new Utility Matrix. Test your predictions against the test set and find out if clustering items can improve your prediction. Write about your findings within 4-5 sentences.

Part 2: Clustering Users and Items
In this part of the project, you will be using the MovieLens dataset to create a utility matrix. Each of the users and items will be defined in the dataset itself. Refer to data/README to find out how the data is structured.

It is difficult to detect the similarity between items and users because we have little information about user-item pairs in this sparse utility matrix. Even if two items belong to the same genre, there are likely to be very few users who bought or rated both. Similarly, even if two users both like a genre or multiple genres, they may not have bought any items to indicate that.

One way of dealing with this lack of data is to cluster items and/or users. In this example, we will cluster similar movies based on their properties. After clustering the items, we can revise the utility matrix so the columns represent clusters of items and the entry for User u and Cluster c is the average rating that u gave to the members of Cluster c that u did not individually rate. Note that u may have rated none of the cluster members, in which case the entry for u and c is left blank.

Table 2. An example utility matrix where the movies have been clustered into two groups - HP and SW Screen Shot 2014-07-23 at 8.54.57 PM.png

Part 2 Instructions:

Begin with part2.py template included in the code template.
We will use the MovieLens Dataset (Training Set, Test Set) for this part, which should also be included in the project template. Further information about the data is in data/README.
Implement/copy in the functions from part1.py, i.e. pcs() and guess(), adapting them for the new dataset as necessary.
Improve your guesses by clustering similar users and/or items (see comments at the bottom of part2.py).

a) You may choose a suitable clustering algorithm/technique. Think about what approach will be most suitable given the kinds of features available to you in the dataset, and how you are planning to use the clustering results.

b) You will still be using Pearson Correlation Similarity Measure to fill the utility matrix; but think of how you can improve your selection of similar users and candidate items using the clusters you find.

Test the ratings you get with the test set. Remember that for each user, you will first have to find out which cluster they belong to and then find their ratings.

a) Train your model (i.e. generate the utility matrix) using the training dataset (u.base).

b) Use the u.test dataset to test your modified implementation with clustering. The u.test file contains user, items pairs and ratings given by them. So you can use the mean squared error metric to verify performance of your recommender system.

c) Once you have a recommender system that performs clustering, submit predictions on the test set here. Submit your answers in the form of user_id, item_id, rating rows in the text box (retaining the order in the file). It will return you a score which measures the Mean Squared Error between your predicted ratings and ground truth data. Your score should be smaller than 1.066, which was returned from a recommendation system using no clustering and a top_n of 150.

(Optional) Make a user interface, on the terminal or otherwise, where you will ask a user to rate 10 movies and then recommend 5 movies they should watch.

Note: ml2_project_extras.zip contains additional file(s) that are relevant but not necessary for completing the project requirements. They have been provided for further self-exploration.

Part 3: Reflections
In this final part of the project, we take a closer look at how you developed your solution.

Answer the following questions reflecting upon your choices and methodology [suggested response lengths are indicated]:

What clustering algorithm/technique have you used to group similar users and/or items? [up to 100 words]
(Optional) Were there other techniques unrelated to clustering that you employed? [up to 150 words]
Did using the clustering algorithm improve the accuracy of your prediction? Can you support your claim with a graph/table showing how results changed? [150 - 250 words]
Give an example of one clustering algorithm which would not be very suitable here (and explain why). [150 - 250 words]
How would you incorporate user feedback to improve recommendations over time? E.g., a user rates a previously unrated item as 5 that your algorithm had predicted a 2 for. Can you think of a way to inform the system with this new fact to help refine future predictions? [200 - 300 words; outline of an algorithm would be ideal]
Can you think of a different application (not recommendation systems) where collaborative filtering could be used, and why? [150 - 250 words]
Submission Instructions
For students in the full course experience, please submit the following:

A PDF Document that includes answers to Part 1, Step 3 and all questions in Part 3. (If you performed Part 1, Step 4, include your findings in your document.) Ensure that your report does not exceed 10 pages (including charts). Any report over ten pages will not be graded.

Create a .zip or .tar.gz of your code for parts 1 and 2. You can also upload your source code somewhere and send us a link to it.

Make a list of Web sites, books, forums, blog posts, github repositories etc that you referred to or used in this submission (Add N/A if you did not use such resources)

Please carefully read the following statement and include it in your email: â€œI hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc. By including this in my email, I understand that I will be expected to explain my work in a video call with a Udacity coach before I can receive my verified certificate.â€

Send items 1-4 above to ml-project@udacity.com. Within 7 days of your submission, you will receive an email from your project evaluator (who will not be the coach youâ€™ve been working with) with a graded rubric and instructions for next steps.

If your project meets expectation, you will have an exit interview with us. The purpose of this interview is to discuss your analysis and verify that you are the person who created it. Donâ€™t worry, this interview will not be difficult or stressful (assuming you are the person who made the analysis :) ).

For any further questions, please review the Udacity Project Submission FAQ or email your coach.
