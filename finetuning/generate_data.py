import json
import random

# Data pools for generation
roles = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "Frontend Developer", "Backend Developer", "Full Stack Engineer", "Machine Learning Engineer"]
seniorities = ["Junior", "Mid-level", "Senior", "Lead", "Staff"]

tech_stacks = {
    "Web": ["React", "Node.js", "TypeScript", "GraphQL", "Next.js", "Tailwind CSS"],
    "Data": ["Python", "Pandas", "PyTorch", "TensorFlow", "SQL", "Spark"],
    "Backend": ["Java", "Spring Boot", "Go", "Docker", "Kubernetes", "Microservices", "AWS"],
    "General": ["Git", "CI/CD", "Agile", "Jira", "Unit Testing"]
}

resume_templates = [
    "Experience with {skill1} and {skill2}. Built a project using {skill3}.",
    "Developed scalable applications using {skill1}. Proficient in {skill2}.",
    "Worked on a team to migrate legacy code to {skill1}. Used {skill2} for testing.",
    "Maintained backend services written in {skill1}. Optimized {skill2} queries.",
    "Led the frontend team in adopting {skill1}. Improved performance with {skill2}.",
]

jd_templates = [
    "We are looking for a {role} with strong experience in {skill1}, {skill2}, and {skill3}.",
    "Must have 3+ years of experience with {skill1} and a solid understanding of {skill2}. Bonus: {skill3}.",
    "Join our team to build the next generation of {skill1} based systems. Requirements: {skill2}, {skill3}.",
    "The ideal candidate is proficient in {skill1} and has deployed applications using {skill2}. Knowledge of {skill3} is required.",
]

analysis_templates = [
    "Gap Analysis:\n- The candidate has strong {skill1} experience but lacks specific mention of {skill3}.\n- The job emphasizes {skill2}, which is only briefly mentioned.\n\nSuggested Bullet Points:\n\\item Designed and implemented {skill3} workflows to enhance data processing efficiency.\n\\item Optimized {skill2} performance, reducing latency by 20%.\n\\item Integrated {skill1} modules into the core architecture.",
    "Gap Analysis:\n- Resume highlights {skill1} well but misses the {skill3} requirement found in the JD.\n- Needs to emphasize {skill2} deployment experience.\n\nSuggested Bullet Points:\n\\item Deployed scalable {skill2} clusters managing high-traffic loads.\n\\item Developed {skill3} integrations to streamline third-party API communication.\n\\item Led {skill1} code reviews ensuring best practices and maintainability.",
    "Gap Analysis:\n- The candidate shows potential in {skill1} but the role demands deep {skill3} knowledge.\n- {skill2} is a critical missing keyword.\n\nSuggested Bullet Points:\n\\item Architected robust {skill3} solutions for real-time data streaming.\n\\item Automated {skill2} pipelines using CI/CD best practices.\n\\item Refactored {skill1} codebase to improve modularity and testability.",
]

def generate_entry():
    role = random.choice(roles)
    seniority = random.choice(seniorities)
    category = random.choice(list(tech_stacks.keys()))
    stack = tech_stacks[category] + tech_stacks["General"]
    
    # Pick 3 random skills for this scenario
    current_skills = random.sample(stack, 3)
    skill1, skill2, skill3 = current_skills
    
    # Generate Resume Snippet
    resume_text = random.choice(resume_templates).format(
        skill1=skill1, 
        skill2=skill2, 
        skill3=random.choice(stack) # Random other skill or same
    )
    
    # Generate JD Snippet
    jd_text = random.choice(jd_templates).format(
        role=f"{seniority} {role}",
        skill1=skill1,
        skill2=skill2,
        skill3=skill3
    )
    
    # Generate Analysis
    # We deliberately assume the "gap" is around skill3 or how skill2 is applied to make it interesting
    analysis_text = random.choice(analysis_templates).format(
        skill1=skill1,
        skill2=skill2,
        skill3=skill3
    )
    
    return {
        "resume_context": resume_text,
        "job_description_context": jd_text,
        "output_analysis": analysis_text
    }

def main():
    dataset = []
    for _ in range(300):
        dataset.append(generate_entry())
        
    with open("finetuning/dataset.jsonl", "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")
            
    print("Generated 300 rows in finetuning/dataset.jsonl")

if __name__ == "__main__":
    main()

