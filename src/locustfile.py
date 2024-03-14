from locust import HttpUser, task, between

class MyUser(HttpUser):
    host = "http://127.0.0.1:5000"  
    wait_time = between(1, 3)

    @task
    def my_post_request(self):
        data = {"question": "what is generated-angular-app in wavemaker"} 
        response = self.client.post("/answer", json=data)
        if response.status_code == 200:
            print("POST request successful")
        else:
            print(f"POST request failed with status code {response.status_code}")


