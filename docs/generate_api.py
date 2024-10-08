"""Generate the code reference pages and navigation."""
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

project_dir = "cd_v_partition"
reference_dir = "reference"

for path in sorted(Path(project_dir).rglob("**/*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to(project_dir).with_suffix(".md")
    full_doc_path = Path(reference_dir, doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    match len(parts):
        case 1:
            nav_parts = parts
        case 2:
            nav_parts = (parts[1],)
        case _:
            nav_parts = tuple([parts[1]] + [p.split(".")[-1] for p in parts[2:]])

    nav[nav_parts] = doc_path.as_posix()
    print(f"{parts=}")
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {parts[-1]}\n\n::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(f"{reference_dir}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
