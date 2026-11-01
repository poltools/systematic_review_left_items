

prompt= """
You are an expert in gender studies and political psychology. Your task is to classify the attitude toward feminism and gender equality expressed in a given statement.

For each statement, return:

1. **Main Category** (choose one from the taxonomy below)  
2. **Subcategory** (select the best-fitting one from within that category)  
3. **Justification** (1-2 sentences explaining your reasoning)

---

### TAXONOMY

**1. Workplace & Economic Equality**
- Pay Equity: Belief in equal pay for equal work or critiques of wage gaps.
- Workplace Discrimination: Mentions bias, barriers to promotion, or unfair treatment at work.
- Affirmative Action / Quotas: Support or opposition to reparative policies.
- Work-Life Balance & Childcare: Views on maternity leave, daycare, or role strain.

**2. Gender Roles & Domesticity**
- Traditional Roles: Belief that women's primary duty is homemaking and caregiving.
- Egalitarian Roles: Support for equal division of labor and decision-making in families.
- Parenting & Marriage Norms: Attitudes about who should have authority or caregiving responsibility.
- Role Reversal Acceptance: Comfort with men in caregiving or women as providers.

**3. Feminist Identity & Movement Engagement**
- Self-Identification as Feminist: Expresses personal alignment with feminism.
- Movement Participation: Protests, petitions, donations, or activism for women's rights.
- Feminist Solidarity: Feels affinity with feminist individuals or groups.
- Disidentification / Ambivalence: Rejects or distances self from feminism.

**4. Political & Civic Representation**
- Support for Representation: Advocates for more women in leadership, politics, or public roles.
- Opposition to Representation: Asserts men are better suited for leadership or politics.

**5. Structural & Systemic Critique**
- Patriarchy Critique: Identifies male dominance as a root cause of women's inequality.
- Intersectionality: Mentions race, class, or sexuality in relation to gender issues.
- Capitalism & Gender: Connects women's oppression to economic systems.
- Institutional Reform: Proposes changes to laws, media, religion, or education.

**6. Reproductive Rights & Sexual Autonomy**
- Abortion Rights: Support for or opposition to a woman's right to choose.
- Sexual Norms & Double Standards: Addresses gendered expectations around sex or courtship.
- Autonomy in Relationships: Advocates for women's independence in romantic or sexual decisions.

**7. Gender Stereotypes & Cultural Norms**
- Media & Beauty Standards: Concerns about portrayal of women in culture and media.
- Language & Respectability Norms: Views on how women should speak or act.
- Essentialist Beliefs: Claims about “natural” differences between men and women.

**8. Anti-Feminist / Backlash Attitudes**
- Dismissal of Inequality Claims: Argues that women are not oppressed.
- Feminism Gone Too Far: Believes feminist demands are excessive or unnecessary.
- Equality Already Achieved: Asserts that gender parity already exists.

---

### EXAMPLES

**Statement**: "Women and men should be paid equally for the same work."  
→ Main Category: Workplace & Economic Equality  
→ Subcategory: Pay Equity  
→ Justification: This expresses clear support for wage equality between genders.

**Statement**: "A woman's main role is to be a good wife and mother."  
→ Main Category: Gender Roles & Domesticity  
→ Subcategory: Traditional Roles  
→ Justification: The speaker affirms a traditional view of a woman's domestic responsibilities.

**Statement**: "I attend rallies supporting women's rights whenever possible."  
→ Main Category: Feminist Identity & Movement Engagement  
→ Subcategory: Movement Participation  
→ Justification: The speaker describes civic activism on behalf of feminist causes.

---

### NOW CLASSIFY:

**Statement**: <SEED>

"""
