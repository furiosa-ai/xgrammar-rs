use std::collections::HashSet;

use hf_hub::RepoType;
use xgrammar::huggingface_hub::{
    HuggingfaceError, Params, Repo, compile_glob_pattern, snapshot_download,
};

#[allow(clippy::unnecessary_to_owned)]
fn assert_snapshot_download(
    repo: Repo,
    options: Option<Params>,
    expected_files: &[&str],
) -> Result<(), HuggingfaceError> {
    let tmpdir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("HF_HOME", tmpdir.path());
        std::env::set_var("HF_HUB_CACHE", tmpdir.path().join("hub"));
    }

    let download_path = snapshot_download(repo, options)?;

    // Check all files of expected exist in the path
    let expected_files: HashSet<String> =
        HashSet::from_iter(expected_files.iter().map(|s| s.to_string()));

    let mut actual_files: HashSet<String> = HashSet::new();
    for entry in std::fs::read_dir(download_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            actual_files.insert(file_name.to_string());
        }
    }

    // Check if all expected files are present
    for expected_file in expected_files.iter() {
        assert!(
            actual_files.contains(&expected_file.to_string()),
            "Expected file {} not found",
            expected_file
        );
    }
    // Check if there are no unexpected files
    for actual_file in &actual_files {
        assert!(
            expected_files.contains(actual_file.as_str()),
            "Unexpected file {} found",
            actual_file
        );
    }

    Ok(())
}

#[ignore = "Ignored to mitigate HF download limit issue. Please enable it when you need to test."]
#[test]
fn test_snapshot_download_with_revision() -> Result<(), HuggingfaceError> {
    let repo = Repo::with_revision(
        "google-t5/t5-small".to_string(),
        RepoType::Model,
        "c9c2c8f7fe6aa9ce37f61418d82a01d25cfac393".to_string(),
    );
    let expected = [".gitattributes", "config.json"];

    assert_snapshot_download(repo, None, &expected)
}

#[ignore = "Ignored to mitigate HF download limit issue. Please enable it when you need to test."]
#[test]
fn test_snapshot_download_with_allow_patterns() -> Result<(), HuggingfaceError> {
    let repo = Repo::new("google-t5/t5-small".to_string(), RepoType::Model);
    let expected = [
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    let filters = Params {
        allow_patterns: Some(compile_glob_pattern(&expected).unwrap()),
        ..Default::default()
    };
    assert_snapshot_download(repo, Some(filters), &expected)
}

#[ignore = "Ignored to mitigate HF download limit issue. Please enable it when you need to test."]
#[test]
fn test_snapshot_download_with_ignore_patterns() -> Result<(), HuggingfaceError> {
    let repo = Repo::new("google-t5/t5-small".to_string(), RepoType::Model);
    let filters = Params {
        ignore_patterns: Some(
            compile_glob_pattern(&["*.ot", "onnx/*", "*.msgpack", "*.h5", "spiece.model", "*.bin"])
                .unwrap(),
        ),
        ..Default::default()
    };
    let expected = [
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    assert_snapshot_download(repo, Some(filters), &expected)
}
