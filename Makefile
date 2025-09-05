install-cargo-deps:
	rustup component add clippy rustfmt
	curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
	cargo binstall -y cargo-nextest cargo-machete cargo-sort

lint:
	cargo machete
	cargo sort --grouped --check --workspace
	cargo fmt --all --check

	# Checking a production build
	cargo -q clippy ${MAYBE_RELEASE_FLAG} --all-targets \
		--workspace --locked \
		-- -D warnings

do-lint:
	cargo sort --grouped --workspace
	cargo fmt --all

test:
	cargo nextest run $(TEST_ARGS)


