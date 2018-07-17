from tweepy import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

#Custom stream listener class to collect relevant details
class StreamListener(StreamListener):
    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            return
        print(status.text)
    def on_error(self, status_code):
        if status_code == 420:
            return False

#Since it wasn't explicitly mentioned, keeping the API keys as it is
#API keys
ckey = ''
csec = ''
atok = ''
asec = ''

#Auth
auth = OAuthHandler(ckey, csec)
auth.set_access_token(atok, asec)
api = API(auth)

#Query to search for tweets using emoticons values
#For example, \U0001F602 refers to 'happy'
#1F602 = happy, 1F62D = sad, 1F621= angry, 2764 = love, 1F61C = playful, 1F631 = confused
query = [u'\U0001F602']

#stream prints out the tweets matching the query to the terminal
stream = Stream(auth = api.auth, listener = StreamListener())
stream.filter(track = query, languages = ["en"], stall_warnings = True, async = True)
