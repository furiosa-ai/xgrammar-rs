install-cargo-deps:
	cargo install cargo-sort@2.0.1 cargo-sort

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


