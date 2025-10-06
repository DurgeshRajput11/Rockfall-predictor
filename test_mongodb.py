from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

username = quote_plus("dsgaur112")
password = quote_plus("admin@123")
uri = f"mongodb+srv://{username}:{password}@cluster0.9kwyw2n.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    # Check collections in the database
    db = client["Durgesh_Rajput"]
    print("Collections in 'Durgesh_Rajput':", db.list_collection_names())
    # Check document count in RockfallData
    if "RockfallData" in db.list_collection_names():
        count = db["RockfallData"].count_documents({})
        print(f"Document count in 'RockfallData': {count}")
    else:
        print("Collection 'RockfallData' does not exist.")
except Exception as e:
    print(e)