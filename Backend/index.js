import dotenv from "dotenv";
import express from "express";
import OpenAI from "openai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Initialize OpenAI client
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.DEEPSEEK_API_KEY,
});

// Health check endpoint
app.get("/", (req, res) => {
  res.json({ status: "API is running" });
});

// Chat completion endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const { message, systemPrompt } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const messages = [
      { 
        role: "system", 
        content: systemPrompt || "You are a helpful assistant." 
      },
      { 
        role: "user", 
        content: message 
      }
    ];

    const completion = await openai.chat.completions.create({
      messages,
      model: "deepseek/deepseek-r1-0528-qwen3-8b:free",
    });

    res.json({
      success: true,
      response: completion.choices[0].message.content,
      usage: completion.usage
    });

  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Streaming chat endpoint (optional)
app.post("/api/chat/stream", async (req, res) => {
  try {
    const { message, systemPrompt } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const messages = [
      { 
        role: "system", 
        content: systemPrompt || "You are a helpful assistant." 
      },
      { 
        role: "user", 
        content: message 
      }
    ];




    // Set headers for streaming
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const stream = await openai.chat.completions.create({
      messages,
      model: "deepseek/deepseek-r1-0528-qwen3-8b:free",
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        res.write(`data: ${JSON.stringify({ content })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

    // Recommendation endpoint
    app.post("/api/recommend", async (req, res) => {
        try {
          const studentData = req.body;
      
          if (!studentData) {
            return res.status(400).json({ error: "Student data is required" });
          }
      
          // Build a structured educational prompt
          const prompt = `
      You are an intelligent educational recommender system.
      Given this student profile: ${JSON.stringify(studentData)},
      recommend 3 personalized online courses or learning paths.
      Each recommendation must include:
      - "course": course name
      - "reason": why it fits this student
      - "category": subject or skill area
      
      Return valid JSON array only.
      `;
      
          const completion = await openai.chat.completions.create({
            model: "deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages: [
              { role: "system", content: "You are an educational AI assistant." },
              { role: "user", content: prompt },
            ],
          });
      
          // Parse model response
          const text = completion.choices[0].message.content;
          const clean = text.replace(/```json|```/g, "").trim();
          const recommendations = JSON.parse(clean);
      
          // Add simple explainability weights (XAI simulation)
          const explanation = {
            feature_importance: {
              weaknesses: 0.4,
              interests: 0.35,
              completed_courses: 0.15,
              gpa: 0.1,
            },
          };
      
          res.json({
            success: true,
            recommendations,
            explanation,
            usage: completion.usage,
          });
      
        } catch (error) {
          console.error("Error:", error);
          res.status(500).json({ success: false, error: error.message });
        }
      });
      

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});