from pathlib import Path



directory_in_str=os.getcwd()+'\images'

pathlist = Path(directory_in_str).glob('**/*.jpg')
for path in pathlist:
     # because path is object not string
     path_in_str = str(path)
     print(path_in_str)