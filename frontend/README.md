# Commercial Court Research UI

Day 16 live workspace for the Indian Commercial Court legal research system. It
is a React, Vite, and TypeScript app with a desktop-first claim, citation,
evidence, and verification layout. The primary workflow calls backend
`POST /research`; mocks remain isolated test fixtures and are never a runtime
fallback.

Start the backend on port 8000, then optionally create `.env.local` here:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Never put `GEMINI_API_KEY` in a `VITE_` variable; Vite variables are shipped to
the browser bundle.

```powershell
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Run the verification gates with:

```powershell
npm test
npm run lint
npm run build
```
