import os 
import shutil


root_dir = r"D:\GitHub\SimpleCVPaperAbstractReading\md"

for item in os.listdir(root_dir):
    full_item = os.path.join(root_dir, item)
    if os.path.isfile(full_item):
        pass 
    elif os.path.isdir(full_item):
        for new_item in os.listdir(full_item):
            if new_item.endswith("assets"):
                full = os.path.join(full_item, new_item)
                print(new_item, len(os.listdir(full)), full)

                if len(os.listdir(full)) == 0:
                    shutil.rmtree(full)
