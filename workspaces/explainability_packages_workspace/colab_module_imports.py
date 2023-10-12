#python==3.11.5
#ipykernel==6.25.2
# pytorch==2.0.1
# transformers==4.33.2
# bertviz==1.4.0
# ipywidgets==8.1.1
# captum==0.6.0
# seaborn==0.12.2
# accelerate>=0.20.1
# evaluate==0.4.0


import os

# os.system('pip install transformers')
# os.system('pip install bertviz')
# os.system('pip install captum')
# os.system('pip install accelerate')
# os.system('pip install evaluate')

os.system('pip install -U transformers[torch]')
os.system('pip install -U bertviz')
os.system('pip install -U captum')
os.system('pip install -U accelerate')
os.system('pip install -U evaluate')
os.system('pip install -U torch-xla')

