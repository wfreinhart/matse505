import argparse
import sys
import os

# Import our scripts
sys.path.append(os.path.dirname(__file__))
import migrate_content
import compile_modules
import build_lecture
import generate_transitions
import interpret_lecture

def main():
    parser = argparse.ArgumentParser(description="Content Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Migrate
    parser_migrate = subparsers.add_parser("migrate", help="Migrate content from legacy lectures to new format")

    # Compile
    parser_compile = subparsers.add_parser("compile", help="Compile source modules to JSON")
    parser_compile.add_argument("--watch", action="store_true", help="Watch for changes")

    # Enrich: generate LLM transitions and illustrations, cache in sidecar YAML
    parser_enrich = subparsers.add_parser(
        "enrich",
        help="Generate LLM transitions and illustrations for a lecture (cached in .transitions.yaml)"
    )
    group_enrich = parser_enrich.add_mutually_exclusive_group(required=True)
    group_enrich.add_argument("--lecture", help="Specific lecture ID (e.g. HSDemo_Regression)")
    group_enrich.add_argument("--all", action="store_true", help="Enrich all lectures")
    parser_enrich.add_argument("--force", action="store_true", help="Regenerate even if cached")
    parser_enrich.add_argument(
        "--model", default=generate_transitions.DEFAULT_MODEL,
        help=f"Text model slug (default: {generate_transitions.DEFAULT_MODEL})"
    )
    parser_enrich.add_argument(
        "--image-model", default=generate_transitions.IMAGE_MODEL,
        help=f"Image model slug (default: {generate_transitions.IMAGE_MODEL})"
    )

    # Interpret: append LLM takeaways cell to an already-executed notebook
    parser_interp = subparsers.add_parser(
        "interpret",
        help="Append Key Takeaways cell to executed notebooks (run after nbconvert --execute)"
    )
    group_interp = parser_interp.add_mutually_exclusive_group(required=True)
    group_interp.add_argument("--lecture", help="Specific lecture ID")
    group_interp.add_argument("--all", action="store_true", help="Interpret all lectures")
    parser_interp.add_argument("--force", action="store_true", help="Regenerate even if cached")
    parser_interp.add_argument(
        "--model", default=interpret_lecture.DEFAULT_MODEL,
        help=f"LiteLLM model slug (default: {interpret_lecture.DEFAULT_MODEL})"
    )

    # Build
    parser_build = subparsers.add_parser("build", help="Build lectures from definitions")
    parser_build.add_argument("--lecture", help="Specific lecture ID")
    parser_build.add_argument("--all", action="store_true", help="Build all lectures")
    parser_build.add_argument(
        "--enrich", action="store_true",
        help="Inject LLM transitions and illustrations (auto-generates sidecar if missing)"
    )

    args = parser.parse_args()

    if args.command == "migrate":
        migrate_content.migrate()

    elif args.command == "compile":
        compile_modules.run(watch=args.watch)

    elif args.command == "enrich":
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        image_model = getattr(args, 'image_model', generate_transitions.IMAGE_MODEL)
        if args.lecture:
            generate_transitions.process_lecture(
                args.lecture, force=args.force, model=args.model, image_model=image_model
            )
        else:
            from pathlib import Path
            defs = sorted(Path("lecture_defs").glob("*.yaml"))
            for d in defs:
                if ".transitions" not in d.stem:
                    generate_transitions.process_lecture(
                        d.stem, force=args.force, model=args.model, image_model=image_model
                    )

    elif args.command == "interpret":
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        if args.lecture:
            interpret_lecture.process_lecture(args.lecture, force=args.force, model=args.model)
        else:
            from pathlib import Path
            defs = sorted(Path("lecture_defs").glob("*.yaml"))
            for d in defs:
                if ".transitions" not in d.stem:
                    interpret_lecture.process_lecture(d.stem, force=args.force, model=args.model)

    elif args.command == "build":
        if not args.lecture and not args.all:
            parser_build.print_help()
        else:
            build_lecture.run(lecture=args.lecture, build_all=args.all, enrich=args.enrich)

if __name__ == "__main__":
    main()
