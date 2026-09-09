# Economics 880 Homework

This repository contains our homework assignments and associated code for Economics 880.

## Repository Structure

Each problem set is contained in its own `homework-X` directory. Each directory contains the original homework PDF and all code associated with that assignment.

```text
economics-880-homework/
│
├── readme.md
│
├── tex-items
│   └── preamble.tex
│
├── homework-1/
│   ├── PS1CompF26.pdf
│   ├── deterministic_functions.jl
│   ├── deterministic_grid.jl
│   ├── parallel_functions.jl
│   ├── parallel_script.jl
│   └── tex
│        ├── economics-880-PS1.tex
│        └── ...
│
├── homework-2/
│   ├── PS2CompF26.pdf
│   └── ...
│
├── homework-3/
│   ├── PS3CompF26.pdf
│   └── ...
│
└── ...
```

The `main` branch contains the final, agreed-upon version of each homework assignment. Development should be done on separate branches rather than directly on `main`.

## Git Workflow

**NOTE**: In what follows, I provide git commands for the CLI, but I think its easier to work with git through the Git plugin in VSCode, or a simialr utility through your preffered IDE.

For each homework, we can create separate branches from `main`. This allows us to work independently, compare our solutions, and decide which changes to merge into the final submission.

### Branch Naming

For Homework 1:

```text
aarons_branch_1
arthurs_branch_1
```

For Homework 2:

```text
aarons_branch_2
arthurs_branch_2
```

and so on.

## Starting a New Homework

First, make sure the local `main` branch is up to date:

```bash
git switch main
git pull --rebase origin main
```

Then create your branch for the homework, for example:

```bash
git switch -c arthurs_branch_1
```

Each branch starts from the same version of `main`.

```text
                         aarons_branch_1
                        /
main ------------------<
                        \
                         arthurs_branch_1
```

## Working on a Branch

Make changes normally and periodically commit them:

```bash
git status
git add .
git commit -m "Implement deterministic solution"
```

The first time you push a new branch:

```bash
git push -u origin aarons_branch_1
```

After that:

```bash
git push
```

## Comparing and Combining Work

Once both solutions are ready, we compare the two branches and decide what should become the final solution.

```text
                         aarons_branch_1
                        /               \
main ------------------<                 >---- final solution
                        \               /
                         arthurs_branch_1
```


## Merging into `main`

Once we have agreed on the final version, switch back to `main` and make sure it is current:

```bash
git switch main
git pull origin main
```

Then merge the branch containing the agreed-upon solution. For example:

```bash
git merge aarons_branch_1
git push origin main
```

If changes from both branches are needed, they should be reconciled before the final submission.

## Final Submission

The `main` branch should always contain the version we intend to submit.

Before submitting:

```bash
git switch main
git pull origin main
git status
```

The overall workflow is:

```text
               ┌── aarons_branch_X ────┐
               │                       │
main ──────────┤                       ├── review/combine ──> main ──> submit
               │                       │
               └── arthurs_branch_X ───┘
```

## General Rules

- Do not do substantial development directly on `main`.
- Start each homework branch from an up-to-date `main`.
- Aaron works on his branch (e.g, `aarons_branch_X`, or some other obvious name), and similary for Arthur
- Commit changes regularly with descriptive commit messages.
- Push personal branches to GitHub so that both people can inspect them.
- Compare both branches before deciding on the final solution.
- Merge only the agreed-upon solution into `main`.
- Keep `main` in a submission-ready state.