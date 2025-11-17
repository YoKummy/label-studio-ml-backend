import os

folder = r'C:\Users\1003380\lgp'  # change this

for filename in os.listdir(folder):
    if ' ' in filename:
        new_name = filename.replace(' ', '_')
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        print(f'Renaming "{filename}" to "{new_name}"')
        os.rename(src, dst)
