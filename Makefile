# This is a suggestion for a Makefile. This assumes you have executed,
#
#     git submodule add git@github.com:entangled/bootstrap-submodule bootstrap
#
# and that you have your literate sources in `./lit`.
#
# Make sure you have the following things installed:
#
#   - Entangled (the daemon)
#   - entangled-filters (the pandoc filters: pip install ...)
#   - Pandoc
#   - BrowserSync (npm install -g ...)
#   - InotifyTools (available from most GNU/Linux distributions)
#
# The website will be built in `./docs`, from which it can be served as
# github.io pages.
#
#
# You should list the Markdown sources here in the order that they should
# appear.
input_files := lit/index.md lit/random-fields.md

# Arguments to Pandoc; these are reasonable defaults
pandoc_args += --template bootstrap/template.html
pandoc_args += --css css/mods.css
pandoc_args += -t html5 -s --mathjax --toc
pandoc_args += --toc-depth 1
pandoc_args += --filter pandoc-bootstrap
pandoc_args += --filter pandoc-eqnos
pandoc_args += --filter pandoc-fignos
pandoc_args += --citeproc
pandoc_args += -f markdown+multiline_tables+simple_tables+citations

# Load syntax definitions for languages that are not supported
# by default. These XML files are in the format of the Kate editor.
pandoc_args += --syntax-definition bootstrap/elm.xml
pandoc_args += --syntax-definition bootstrap/pure.xml
pandoc_args += --highlight-style tango

# Any file in the `lit` directory that is not a Markdown source
# is to be copied to the `docs` directory
static_files := $(shell find -L lit -type f -not -name '*.md')
static_targets := $(static_files:lit/%=docs/%)

.PHONY: site clean watch watch-pandoc watch-browser-sync

# This should build everything needed to generate your web site. That includes
# possible Javascript targets that may need compiling.
site: docs/index.html docs/css/mods.css $(static_targets)

clean:
	rm -rf docs

# Starts a tmux with Entangled, Browser-sync and an Inotify loop for running
# Pandoc.
watch:
	@tmux new-session make --no-print-directory watch-pandoc \; \
		split-window -v make --no-print-directory watch-browser-sync \; \
		split-window -v entangled daemon \; \
		select-layout even-vertical \;

watch-pandoc:
	@while true; do \
		inotifywait -e close_write bootstrap lit Makefile; \
		make site; \
	done

watch-browser-sync:
	browser-sync start -w -s docs

docs/index.html: $(input_files) Makefile
	@mkdir -p docs
	pandoc $(pandoc_args) $(input_files) -o $@

docs/css/mods.css: bootstrap/mods.css
	@mkdir -p docs/css
	cp $< $@

$(static_targets): docs/%: lit/%
	@mkdir -p $(dir $@)
	cp $< $@

