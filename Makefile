lint:
	cargo machete
	cargo sort --grouped --check --workspace
	cargo fmt --all --check

	# Checking a production build
	cargo -q clippy ${MAYBE_RELEASE_FLAG} --all-targets \
		--workspace --locked --features rngd \
		-- -D warnings

do-lint:
	cargo sort --grouped --workspace
	cargo fmt --all
