prompt = """
You are an expert in environmental psychology. Your task is to classify the environmental attitude expressed in a given statement.

For each statement, return:

1. **Main Category** (choose one from the taxonomy below)
2. **Subcategory** (select the best-fitting one from within that category)
3. **Justification** (1-2 sentences explaining your reasoning)

---

### TAXONOMY

**Nature & Human Relationship**
- Biocentrism/Ecocentrism: Nature has intrinsic value; all species have equal rights.
- Anthropocentrism: Human needs come first; nature is a resource.
- Spiritual Stewardship: Nature is sacred; humans have a religious/moral duty to protect it.

**Technology & Development**
- Techno-Optimism: Technology can solve environmental problems.
- Techno-Skepticism: Technology causes harm or alienation.
- Urbanization/Infrastructure: Views on housing, transport, or land development.

**Policy & Governance**
- Regulatory Support: Pro-environmental laws, taxes, or rules.
- Government Skepticism: Opposition to environmental regulations.
- Internationalism: Global treaties, cooperation, or UN involvement.
- Environmental Justice: Fairness, marginalized communities, or equitable impacts.

**Personal Identity & Responsibility**
- Environmental Identity: Sees self as an environmentalist.
- Personal Behavior: Mentions recycling, conserving, or consumer habits.
- Activism & Engagement: Protests, donations, organizing, or civic action.

**Environmental Threat Perception**
- Climate Concern: Belief in climate urgency or risk.
- Climate Denial/Skepticism: Minimizes environmental threats.
- Pollution Concern: Air, water, or industrial pollution.
- Resource Scarcity: Concern about depletion of water, land, energy, etc.

**Consumption & Materialism**
- Eco-Consumerism: Favors eco-friendly or green products.
- Anti-Consumerism: Critiques materialism; supports simplicity.

**Intergenerational & Future Orientation**
- Future Generations: Motivated by the wellbeing of future descendants.
- Sustainability Ethics: Supports long-term ecological balance over short-term gains.

---

### EXAMPLES

**Statement**: "We must protect forests, even if it means sacrificing economic development."  
→ Main Category: Nature & Human Relationship  
→ Subcategory: Biocentrism/Ecocentrism  
→ Justification: The speaker values forests independently of human benefit, implying intrinsic worth of nature.

---

**Statement**: "Technology will help us solve climate change faster than any policy ever could."  
→ Main Category: Technology & Development  
→ Subcategory: Techno-Optimism  
→ Justification: The statement expresses strong faith in technological solutions to environmental problems.

---

**Statement**: "I always switch off the lights when I leave the room because I care about energy conservation."  
→ Main Category: Personal Identity & Responsibility  
→ Subcategory: Personal Behavior  
→ Justification: The speaker describes an individual action that reflects environmental consciousness.

---

### NOW CLASSIFY:

**Statement**: <SEED>

"""