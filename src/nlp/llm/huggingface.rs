use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Error;

#[derive(Deserialize, Serialize, Debug)]
pub struct GeneratedText {
    pub generated_text: String,
}

pub struct HuggingFaceModel {
    api_url: String,
    headers: HeaderMap,
}

impl HuggingFaceModel {
    pub fn new(api_url: &str, api_key: &str, model_id: &str) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );

        let api_url = format!("{}/{}", api_url, model_id);

        HuggingFaceModel {
            api_url,
            headers,
        }
    }

    pub fn generate_text(&self, prompt: &str, parameters: HashMap<String, serde_json::Value>) -> Result<String, Error> {
        let client = Client::new();
        let mut payload = parameters;
        payload.insert("inputs".to_string(), serde_json::Value::String(prompt.to_string()));

        let response = client.post(&self.api_url)
            .headers(self.headers.clone())
            .json(&payload)
            .send()?;

        let status = response.status();
        let response_text = response.text()?;

        if !status.is_success() {
            eprintln!("Request failed with status: {}", status);
            eprintln!("Response: {}", response_text);
            return Ok(response_text);
        }

        Ok(response_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use serde_json::Value;

    #[test]
    fn test_generate_text() {
        let api_url = "https://api-inference.huggingface.co/models";
        let api_key = "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";
        let model_id = "mistralai/Mistral-7B-Instruct-v0.2";

        let hf_model = HuggingFaceModel::new(api_url, api_key, model_id);

        let prompt = "Translate English to French: 'Hello, how are you?'";
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), serde_json::json!(0.7));
        params.insert("max_new_tokens".to_string(), serde_json::json!(250));

        match hf_model.generate_text(prompt, params) {
            Ok(response) => {
                println!("Response: {}", response);
                let parsed: Vec<Value> = serde_json::from_str(&response).expect("Failed to parse JSON response");
                if let Some(generated_text) = parsed.get(0).and_then(|v| v.get("generated_text")).and_then(|v| v.as_str()) {
                    println!("Generated Text: {}", generated_text);
                    assert!(generated_text.contains("Bonjour"), "Response should contain the French translation 'Bonjour'");
                } else {
                    panic!("Response did not contain 'generated_text'");
                }
            },
            Err(e) => panic!("Error: {}", e),
        }
    }
}
