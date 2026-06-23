import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("MobileCollection")

results = collection.query.fetch_objects(limit=3)

for obj in results.objects:
    print(obj.properties)

client.close()