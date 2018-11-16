import requests
import zipfile

def download(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(r'D:/vizdoom/AI/src', "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def prepare():
	file_id = '1dCvAPPBZk6gvRBIma9AJbpbjZMDSU4M3'
	destination = 'D:/vizdoom/AI/src'
	download(file_id, destination)    
	zip_ref = zipfile.ZipFile(destination, 'r')
	zip_ref.extractall(destination)
	zip_ref.close()

if __name__ == "__main__":  
	prepare()