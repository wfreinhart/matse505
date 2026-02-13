import argparse
import sys
import os

# Import our scripts
sys.path.append(os.path.dirname(__file__))
import migrate_content
import compile_modules
import build_lecture

def main():
    parser = argparse.ArgumentParser(description="Content Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Migrate
    parser_migrate = subparsers.add_parser("migrate", help="Migrate content from legacy lectures to new format")

    # Compile
    parser_compile = subparsers.add_parser("compile", help="Compile source modules to JSON")
    parser_compile.add_argument("--watch", action="store_true", help="Watch for changes")

    # Build
    parser_build = subparsers.add_parser("build", help="Build lectures from definitions")
    parser_build.add_argument("--lecture", help="Specific lecture ID")
    parser_build.add_argument("--all", action="store_true", help="Build all lectures")

    args = parser.parse_args()

    if args.command == "migrate":
        migrate_content.migrate()
    elif args.command == "compile":
        compile_modules.run(watch=args.watch)
    elif args.command == "build":
        if not args.lecture and not args.all:
            parser_build.print_help()
        else:
            build_lecture.run(lecture=args.lecture, build_all=args.all)

if __name__ == "__main__":
    main()
