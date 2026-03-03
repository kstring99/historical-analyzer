# Email Draft — Glenn (ESA Technical Director)

**Subject: Historical Documentation Analyzer — Integration Questions**

Hi Glenn,

I built a tool that automates the historical documentation sections of Phase I ESAs — aerial photographs, topographic maps, and city directories. It ingests the ERIS PDF packages, runs them through an AI vision model, and outputs formatted tables (Subject Property / Adjoining / Surrounding) with year ranges, Issues Noted, and observation descriptions that match our Quire template format. It also generates a cross-referencing summary paragraph that ties findings across all three source types.

It's model-agnostic — works with OpenAI or Anthropic — so it should fit into whatever the company is already using.

I have a few questions to make sure I integrate it cleanly with what you're building:

1. **API/Model** — Is the photo captioning tool using GPT-4o through our enterprise OpenAI subscription, or is it going through Azure OpenAI? Want to make sure I'm hitting the same endpoint.

2. **Hosting** — Where is the photo captioner hosted right now? (Azure, internal server, etc.) I have this packaged as a Docker container so deployment should be straightforward.

3. **Authentication** — How are you handling user auth on the Teams app? I want to match the same pattern.

4. **Repo/Deployment** — Is there a shared repo or deployment pipeline for these tools, or should I just hand you the container?

5. **Website plans** — You mentioned building a website to host everything — any decisions on tech stack yet? Happy to make sure this fits whatever framework you're going with.

I can demo it whenever works for you. It's running locally right now and ready to go.

Kyle
