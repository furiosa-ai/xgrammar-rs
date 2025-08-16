use std::path::PathBuf;

use globset::{Glob, GlobMatcher};
pub use hf_hub::{
    Repo, RepoType,
    api::{
        RepoInfo,
        sync::{Api, ApiBuilder, ApiError, ApiRepo},
    },
};

#[derive(thiserror::Error, Debug)]
pub enum HuggingfaceError {
    #[error("fail to download: {0}")]
    ApiError(#[from] hf_hub::api::sync::ApiError),
    #[error("invalid glob pattern: {0}")]
    InvalidPatternError(#[from] globset::Error),
}

pub fn compile_glob_pattern(patterns: &[&str]) -> Result<Vec<GlobMatcher>, globset::Error> {
    let compiled_patterns = patterns
        .iter()
        .map(|s| Glob::new(s).map(|g| g.compile_matcher()))
        .collect::<Result<Vec<GlobMatcher>, globset::Error>>()?;

    Ok(compiled_patterns)
}

#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    // TODO - add expected model_kind to prevent downloading a large model unnecessarily
    pub allow_patterns: Option<Vec<GlobMatcher>>,
    pub ignore_patterns: Option<Vec<GlobMatcher>>,
}

impl DownloadOptions {
    pub fn is_matched(&self, filename: &str) -> bool {
        // Referred from https://github.com/huggingface/huggingface_hub/blob/a09927331ec0ed2df90968da2200c6bef8ab4117/src/huggingface_hub/utils/_paths.py#L124
        if let Some(patterns) = &self.allow_patterns {
            if !patterns.iter().any(|glob| glob.is_match(filename)) {
                return false;
            }
        }

        if let Some(patterns) = &self.ignore_patterns {
            if patterns.iter().any(|glob| glob.is_match(filename)) {
                return false;
            }
        }

        true
    }
}

pub fn snapshot_download(
    repo: Repo,
    options: Option<DownloadOptions>,
) -> Result<PathBuf, ApiError> {
    let api: Api = ApiBuilder::from_env().build()?;
    let api_repo: ApiRepo = api.repo(repo.clone());
    let repo_info: RepoInfo = api_repo.info()?;

    for sibling in repo_info.siblings {
        if let Some(options) = &options {
            if !options.is_matched(&sibling.rfilename) {
                continue;
            }
        }
        api_repo.get(&sibling.rfilename)?;
    }

    let config_json_path: PathBuf = api_repo.get("config.json")?;
    config_json_path
        .parent()
        .ok_or(ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Parent directory not found", // if cache directory is root directory
        )))
        .map(PathBuf::from)
}
