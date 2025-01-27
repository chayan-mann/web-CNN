
function About() {
  return (
    <div className="pt-24 flex flex-col items-center justify-center min-h-screen px-6 bg-gray-900 text-white">
      <div className="bg-gray-800 shadow-lg rounded-lg p-8 max-w-4xl">
        {/* <h1 className="text-4xl font-bold text-center mb-8 text-blue-400">
          About Brain Tumors
        </h1> */}
        <div className="flex flex-col md:flex-row items-center mb-8">
          <img
            src="https://via.placeholder.com/300x200"
            alt="Brain Scan"
            className="rounded-lg shadow-lg md:w-1/2 mb-6 md:mb-0 md:mr-6"
          />
          <div className="md:w-1/2">
            <p className="text-lg leading-relaxed text-gray-300">
              A brain tumor is an abnormal growth of cells in the brain or spinal
              cord. Brain tumors can be benign (non-cancerous) or malignant
              (cancerous). These tumors disrupt normal brain function by
              compressing, invading, or causing swelling in the surrounding
              tissues.
            </p>
          </div>
        </div>

        <h2 className="text-2xl font-semibold mb-4 text-blue-300">
          Common Symptoms
        </h2>
        <ul className="list-disc list-inside mb-8 text-gray-300">
          <li>Persistent headaches, often worse in the morning.</li>
          <li>Nausea or vomiting unrelated to other illnesses.</li>
          <li>Seizures or unusual behavior.</li>
          <li>Difficulty with vision, speech, or balance.</li>
          <li>Weakness or numbness in limbs.</li>
        </ul>

        <div className="flex flex-col md:flex-row items-center mb-8">
          <img
            src="https://via.placeholder.com/300x200"
            alt="Treatment Illustration"
            className="rounded-lg shadow-lg md:w-1/2 mb-6 md:mb-0 md:mr-6"
          />
          <div className="md:w-1/2">
            <h2 className="text-2xl font-semibold mb-4 text-blue-300">
              Treatments for Brain Tumors
            </h2>
            <p className="text-lg leading-relaxed text-gray-300">
              Treatment options depend on the type and location of the tumor, as
              well as the patient’s overall health. Common treatments include:
            </p>
            <ul className="list-disc list-inside mt-4 text-gray-300">
              <li>
                <strong>Surgery:</strong> Removing the tumor surgically if
                possible.
              </li>
              <li>
                <strong>Radiation Therapy:</strong> Using high-energy rays to
                target and destroy tumor cells.
              </li>
              <li>
                <strong>Chemotherapy:</strong> Using drugs to kill or slow the
                growth of cancer cells.
              </li>
              <li>
                <strong>Targeted Therapy:</strong> Focuses on specific
                abnormalities in tumor cells.
              </li>
            </ul>
          </div>
        </div>

        <h2 className="text-2xl font-semibold mb-4 text-blue-300">
          Prevention Tips
        </h2>
        <div className="flex flex-col md:flex-row items-center mb-8">
          <div className="md:w-1/2 md:mr-6">
            <p className="text-lg leading-relaxed text-gray-300">
              While not all brain tumors are preventable, you can reduce your
              risk by adopting a healthy lifestyle and minimizing exposure to
              risk factors:
            </p>
            <ul className="list-disc list-inside mt-4 text-gray-300">
              <li>Maintain a balanced diet rich in fruits and vegetables.</li>
              <li>Exercise regularly to improve overall health.</li>
              <li>Limit exposure to radiation and harmful chemicals.</li>
              <li>Avoid smoking and excessive alcohol consumption.</li>
              <li>Regular check-ups for early detection.</li>
            </ul>
          </div>
          <img
            src="https://via.placeholder.com/300x200"
            alt="Healthy Lifestyle"
            className="rounded-lg shadow-lg md:w-1/2 mt-6 md:mt-0"
          />
        </div>

        <h2 className="text-2xl font-semibold mb-4 text-blue-300">
          Did You Know?
        </h2>
        <p className="text-lg leading-relaxed text-gray-300">
          Brain tumors can occur in people of all ages, but the risk increases
          with age. Research and advanced treatments have significantly improved
          survival rates and quality of life for patients. Early diagnosis and
          treatment are key to successful outcomes.
        </p>
      </div>
    </div>
  );
}

export default About;
