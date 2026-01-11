# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

I attempted to create a fresh youtube account, but I received no recommendations. 
I will try to use my personal account next. 

^ Above method did not work, so I will now search for videos instead
I am using key words: debate, worst?, best? 
I will also try to search through most views vs. least views

I hope to discover that the model will determine that videos with 'debate' in the title
with positive words have the least views, while titles with negative words have more views
My model should determine what sort of feedback it gets depending on the title

Searched for videos published from BEFORE January 1st 2020. this is because stats dont show
for new videos

Other analytic API only offers stats for likes and dislikes from a channel not per video

very interesting. The top rated comments lean to supporting the video argument, 
while the most recent comments are against it (for best debate in the world)

api capped out at 100 comments per video unfortunately
api capped videos at 43 
"""
import requests
import json
import html
from textblob import TextBlob
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random

output = open("output", "w")

API_KEY = "AIzaSyAq8nRy1TuM54jlbUOugEym6wYpIMs7wng"
CHANNEL_ID = "UCpVSFCPeppcrt6t4ZlSZPyQ"  # new channel
video_ids = []

QUERY = "debate"

url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&part=snippet&maxResults=1000&type=video&regionCode=US&q={QUERY}&publishedBefore=2020-02-01T00:00:00Z"

response = requests.get(url)
data = json.loads(response.text)

# all_video_stats = np.empty((0,)) # structure: [ (polarity,subjectivity), views, likes, (avg_polarity,avg_subjectivity) ]
all_video_stats = []

#video titles

# for item in data['items'][:50]: #starting with 50
j = 1
for item in data['items'][:100]:
    # print(item['snippet']['title'])
    video_title = item["snippet"]["title"]
    video_title = html.unescape(video_title) #rid of appostraphe
    video_id = item["id"]["videoId"] #important for getting the link AND COMMENTS
    print(f"_________________________________\nVideo {j}")
    print(f"Title: {video_title}, Link: https://www.youtube.com/watch?v={video_id}")
    output.write(f"_________________________________\nVideo {j}\n")
    output.write(f"Title: {video_title}, Link: https://www.youtube.com/watch?v={video_id}")
    
    video_url = f"https://www.googleapis.com/youtube/v3/videos?key={API_KEY}&part=statistics&id={video_id}"
    video_response = requests.get(video_url) #only way to get likes and views data
    video_data = json.loads(video_response.text)
    

    view_count = video_data["items"][0]["statistics"]["viewCount"]
    like_count = video_data["items"][0]["statistics"]["likeCount"]

    print(f"Views: {view_count}, Likes: {like_count}")
    output.write(f"Views: {view_count}, Likes: {like_count}")
    
    blob = TextBlob(video_title)
    print(blob.sentiment)
    
    
    #no comments?
    if "commentCount" not in video_data["items"][0]["statistics"]:
        print("Video has no comments.")
        output.write("Video has no comments.")
        continue
    
    #disabled comments- could this correlate with the title?
    if video_data["items"][0]["statistics"]["commentCount"] == "0":
        print("Video has disabled comments.")
        output.write("Video has disabled comments.")
        continue
    
    #video comments in order of TOP RATED 
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&order=relevance&part=snippet&videoId={video_id}&maxResults=500"
    
    #video comments in order of DATE POSTED
    # comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&part=snippet&videoId={video_id}&maxResults=100"

    comments_response = requests.get(comments_url)
    comments_data = json.loads(comments_response.text)
    
    #calculating comment sentiment analysis
    comment_count = 0
    tot_comment_polarity = 0
    tot_comment_subjectivity = 0
    for item in comments_data['items'][:100]: #tried to get 300 comments- max at 100
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comment = html.unescape(comment).replace("<br>", " ") #getting rid of new line
        if (len(comment) < 2000):
            comment_count += 1
            
            commentblob = TextBlob(comment)
            tot_comment_polarity += commentblob.sentiment.polarity
            tot_comment_subjectivity += commentblob.sentiment.subjectivity
            
            print(comment) #uncomment to view comments
            # output.write(comment)
            print("\n")
    avg_pol = tot_comment_polarity/comment_count
    avg_sub = tot_comment_subjectivity/comment_count        
    print(f"\nNumber of comments: {comment_count}")
    print(f"Average comment polarity: {avg_pol}")
    print(f"Average comment subjectivity: {avg_sub}")
    print(f"Youtube TITLE polarity: {blob.sentiment.polarity}")
    print(f"Youtube TITLE subjectivity: {blob.sentiment.subjectivity}\n\n")
    output.write(f"\nNumber of comments: {comment_count}\nAverage comment polarity: {avg_pol}\nAverage comment subjectivity: {avg_sub}\nYoutube TITLE polarity: {blob.sentiment.polarity}\nYoutube TITLE subjectivity: {blob.sentiment.subjectivity}\n\n")
    
    # video_stats = np.array([(blob.sentiment.polarity, blob.sentiment.subjectivity), np.log(int(view_count)), np.log(int(like_count)), (avg_pol, avg_sub)]) 
    video_stats = [(blob.sentiment.polarity, blob.sentiment.subjectivity), int(view_count), int(like_count), (avg_pol, avg_sub)]

    # all_video_stats = np.append(all_video_stats, video_stats)
    all_video_stats.append(video_stats)
    # structure: [ (polarity,subjectivity), views, likes, (avg_polarity,avg_subjectivity) ]
    j+=1

x = np.empty((0, 6))
y = np.empty((0,))

for i in all_video_stats:
    # find the log for the number of views and likes 
    x_i = np.array([i[0][0], i[0][1], np.log(i[1]), np.log(i[2]), i[3][0], i[3][1]]) 
    x = np.vstack([x, x_i])  # feature vector
    
    y_i = i[3][0]
    y = np.append(y, y_i) #target 
    
model = LinearRegression().fit(x, y)


new_data = [[(0.05, 0.2), 1000000, 5000, (0.2, 0.4)]]  #dummy data
new_X = np.array([[    new_data[0][0][0],
    new_data[0][0][1],
    np.log(new_data[0][1]),
    np.log(new_data[0][2]),
    new_data[0][3][0],
    new_data[0][3][1]
]])
predicted_y = model.predict(new_X)

#returns [-0.07263613] with 5 videos and 10 comments
#returns [0.2] with 50 videos and 10 comments
#returns [0.2] with 50 videos/100 videos and 100 comments
print("Video P:\tAud P:\tPred P:\t Views:\t Likes:\t")

print(f"{new_data[0][0][0]}\t{new_data[0][-1][0]}\t{predicted_y[0]}\t{new_data[0][1]}\t{new_data[0][2]}")

output.write(f"_______________________________\nVideo P:\tAud P:\tPred P:\t Views:\t Likes:\t\n{new_data[0][0][0]}\t{new_data[0][-1][0]}\t{predicted_y[0]}\t{new_data[0][1]}\t{new_data[0][2]}")
i = 0

true = []
pred = []
while (i < 20):
    new_data = [all_video_stats[random.randint(0, len(all_video_stats)-1)]]
    
    if (new_data[0][0][0] < 0):
        new_X = np.array([[    new_data[0][0][0],
            new_data[0][0][1],
            np.log(new_data[0][1]),
            np.log(new_data[0][2]),
            new_data[0][3][0],
            new_data[0][3][1]
        ]])
        predicted_y = model.predict(new_X)
        pred.append(predicted_y)
        true.append(new_data[0][-1][0])
        print(f"{new_data[0][0][0]}\t{new_data[0][-1][0]}\t{predicted_y[0]}\t{new_data[0][1]}\t{new_data[0][2]}")
        output.write(f"{new_data[0][0][0]}\t{new_data[0][-1][0]}\t{predicted_y[0]}\t{new_data[0][1]}\t{new_data[0][2]}")

        # print(f"Predicted audience sentiment: {predicted_y[0]}\nActual audience sentiment: {new_data[0][-1][0]}\nVideo Sentiment: {new_data[0][0][0]}\n")
        i += 1
        
    else:
        continue

mse = mean_squared_error(true, pred)
# mse = round(mse,10)
print(f"Mean Square Error: {mse}")
output.write(f"Mean Square Error: {mse}\nMSE(Rounded): {round(mse,10)}")
