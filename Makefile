install-cargo-deps:
	rustup component add clippy rustfmt
	cargo install cargo-nextest cargo-machete@0.7.0 cargo-sort

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


