# Commercial Court Research UI

Day 15 presentation shell for the Indian Commercial Court legal research
workspace. It is a React, Vite, and TypeScript app with a desktop-first
claim, citation, evidence, and verification layout.

The UI currently pairs clearly labelled, API-shaped `/answer` and `/verify`
static fixtures from `src/mocks/answerResponses.ts`. It does not call the
backend, orchestrate the two endpoints, or write to the corpus.

```powershell
npm install
npm run dev
```

Run the verification gates with:

```powershell
npm test
npm run lint
npm run build
```
