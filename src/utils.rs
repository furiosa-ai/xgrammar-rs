use crate::error::XGrammarErr;

pub fn get_json_field<'a>(
    json: &'a serde_json::Value,
    field: &str,
) -> Result<&'a serde_json::Value, XGrammarErr> {
    json.get(field).ok_or_else(|| XGrammarErr::MissingJsonField(field.to_string()))
}
