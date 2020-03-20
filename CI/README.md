# CI - Continuous Integration

When developing a software, both as an individual and in a team, it is difficult to follow 
all of the standards that you or your team agreed to follow; this includes coding styles,
unit tests, documentation, etc.

To make this easier, one can employ different tools and programming packages to enforce some
if not all of agreements done. These tools are listed here, grouped by the programming language.

## NodeJS

[Husky](https://github.com/typicode/husky). Enables setting [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
inside the project. Git hooks allow executing scripts when a particular git command is executed 
(i.e. one can set `pre-commit` hook to check if the code is correctly formatted and executable
before the `git commit` command is executed, notifying the user of any possible errors in the
code). The whole list of possible git hooks is available [here](https://git-scm.com/docs/githooks).

[ESLint](https://eslint.org/). This library checks if the developed code follows the agreed
coding style. It's configuration can be heavily customized to match the projects requirements.

[Typescript](https://www.typescriptlang.org/). It is a typed superset of JavaScript that compiles 
to plain JavaScript. It is used to find errors in the code before run-time (i.e. identifying which
parts of the code are not compatible with each other). It's benefits can be seen in larger javascript 
projects as it checks if the function received the correct argument types.
[Get started with Typescript in 2019](https://www.robertcooper.me/get-started-with-typescript-in-2019)
is a good starting point to understand what is typescript capable of an how to utilize it.




