from database import Database

db = Database()

db.import_jsonl(
    "intros.jsonl",
    channel_id=1131831736771301398
)

print("Import complete!")

msg = db.get_message(1132341105665527828)

print(msg["author_name"])
print(msg["content"][:100])

db.close()