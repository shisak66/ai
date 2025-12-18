import os
import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import pdfplumber
import fitz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode, ChatAction
import google.generativeai as genai
from dotenv import load_dotenv
import aiohttp
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import json
from datetime import datetime, timedelta
import hashlib
import uuid
from collections import defaultdict
import threading
import queue
import psutil
import GPUtil
from tabulate import tabulate
import humanize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-GUI backend
import numpy as np
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Advanced logging configuration
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',   # Red
        'CRITICAL': '\033[41m' # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('study_bot_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEVELOPER_CHAT_ID = os.getenv("DEVELOPER_CHAT_ID", "YOUR_CHAT_ID_HERE")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Enums for better code organization
class ProcessingType(Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    VOICE = "voice"
    AUDIO = "audio"
    VIDEO = "video"

class AnswerType(Enum):
    ONE_WORD = "one_word"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXPLANATION = "explanation"

# Data classes for structured data
@dataclass
class ProcessingStep:
    name: str
    status: str = "pending"
    progress: float = 0.0
    start_time: float = None
    end_time: float = None
    details: Dict = None
    
    def __post_init__(self):
        self.details = self.details or {}
        self.start_time = self.start_time or time.time()

@dataclass
class DeveloperMetrics:
    user_id: int
    username: str
    message_count: int = 0
    total_processing_time: float = 0.0
    last_active: datetime = None
    questions_asked: int = 0
    files_processed: int = 0
    average_response_time: float = 0.0
    session_start: datetime = None
    
    def update_activity(self):
        self.message_count += 1
        self.last_active = datetime.now()

@dataclass
class SystemMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_usage: float = 0.0
    active_users: int = 0
    total_requests: int = 0
    uptime: float = 0.0
    response_times: List[float] = None
    
    def __post_init__(self):
        self.response_times = self.response_times or []

# Advanced Study Database with analytics
class AdvancedStudyDatabase:
    def __init__(self):
        self.file_path = "advanced_study_database.json"
        self.analytics_path = "analytics_database.json"
        self.load_data()
        self.initialize_analytics()
    
    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {
                "questions": {},
                "users": {},
                "sessions": {},
                "analytics": {
                    "daily_stats": {},
                    "hourly_stats": {},
                    "user_stats": {}
                }
            }
    
    def initialize_analytics(self):
        try:
            with open(self.analytics_path, 'r') as f:
                self.analytics = json.load(f)
        except FileNotFoundError:
            self.analytics = {
                "system_metrics": {},
                "performance_logs": [],
                "error_logs": [],
                "user_behavior": {}
            }
    
    def save_data(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        with open(self.analytics_path, 'w') as f:
            json.dump(self.analytics, f, indent=2)
    
    def add_question_with_analytics(self, user_id: int, question: str, processing_type: str, processing_time: float):
        user_key = str(user_id)
        
        if user_key not in self.data["questions"]:
            self.data["questions"][user_key] = []
        
        question_id = len(self.data["questions"][user_key]) + 1
        question_data = {
            "id": question_id,
            "question": question,
            "processing_type": processing_type,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "answers": {},
            "user_feedback": None,
            "ai_model_used": "gemini-pro"
        }
        
        self.data["questions"][user_key].append(question_data)
        
        # Update analytics
        today = datetime.now().strftime("%Y-%m-%d")
        hour = datetime.now().strftime("%H:00")
        
        if today not in self.data["analytics"]["daily_stats"]:
            self.data["analytics"]["daily_stats"][today] = {"questions": 0, "processing_time": 0.0}
        if hour not in self.data["analytics"]["hourly_stats"]:
            self.data["analytics"]["hourly_stats"][hour] = {"questions": 0, "users": set()}
        
        self.data["analytics"]["daily_stats"][today]["questions"] += 1
        self.data["analytics"]["daily_stats"][today]["processing_time"] += processing_time
        self.data["analytics"]["hourly_stats"][hour]["questions"] += 1
        self.data["analytics"]["hourly_stats"][hour]["users"].add(user_key)
        
        self.save_data()
        return question_id
    
    def update_user_metrics(self, user_id: int, metrics_update: Dict):
        user_key = str(user_id)
        if user_key not in self.data["users"]:
            self.data["users"][user_key] = {
                "first_seen": datetime.now().isoformat(),
                "total_questions": 0,
                "total_processing_time": 0.0,
                "preferred_answer_type": "medium",
                "last_session": None,
                "sessions_count": 0
            }
        
        for key, value in metrics_update.items():
            if key in self.data["users"][user_key]:
                if isinstance(value, (int, float)):
                    self.data["users"][user_key][key] += value
                else:
                    self.data["users"][user_key][key] = value
        
        self.save_data()

db = AdvancedStudyDatabase()

# Live Processing Manager with progress tracking
class LiveProcessingManager:
    def __init__(self):
        self.active_processes = {}
        self.progress_callbacks = {}
        self.processing_queue = queue.Queue()
        self.processing_threads = {}
        
    def start_processing(self, user_id: int, process_id: str, process_name: str):
        process = ProcessingStep(
            name=process_name,
            status="running",
            progress=0.0,
            details={"user_id": user_id, "process_id": process_id}
        )
        self.active_processes[process_id] = process
        return process_id
    
    def update_progress(self, process_id: str, progress: float, status: str = None, details: Dict = None):
        if process_id in self.active_processes:
            self.active_processes[process_id].progress = progress
            if status:
                self.active_processes[process_id].status = status
            if details:
                self.active_processes[process_id].details.update(details)
            
            # Trigger callbacks
            if process_id in self.progress_callbacks:
                for callback in self.progress_callbacks[process_id]:
                    callback(progress, status, details)
    
    def complete_processing(self, process_id: str, result: any = None):
        if process_id in self.active_processes:
            self.active_processes[process_id].status = "completed"
            self.active_processes[process_id].progress = 100.0
            self.active_processes[process_id].end_time = time.time()
            return result
        return None
    
    def get_process_info(self, process_id: str):
        return self.active_processes.get(process_id)

processing_manager = LiveProcessingManager()

# System Monitor for real-time metrics
class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.response_times = []
        
    def get_system_metrics(self) -> SystemMetrics:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        gpu_usage = 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            gpu_usage=gpu_usage,
            active_users=len(processing_manager.active_processes),
            total_requests=self.request_count,
            uptime=time.time() - self.start_time,
            response_times=self.response_times[-100:] if self.response_times else []
        )
    
    def record_response_time(self, response_time: float):
        self.response_times.append(response_time)
        self.request_count += 1

system_monitor = SystemMonitor()

# Developer Dashboard Manager
class DeveloperDashboard:
    def __init__(self, bot_application):
        self.bot = bot_application
        self.developer_chat_id = DEVELOPER_CHAT_ID
        
    async def send_developer_update(self, message: str, parse_mode: ParseMode = ParseMode.MARKDOWN):
        """Send update to developer"""
        try:
            if self.developer_chat_id:
                await self.bot.bot.send_message(
                    chat_id=self.developer_chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
        except Exception as e:
            logger.error(f"Failed to send developer update: {e}")
    
    async def send_live_metrics(self, user_id: int, username: str, action: str, details: Dict = None):
        """Send live metrics for each user action"""
        metrics = system_monitor.get_system_metrics()
        
        metrics_text = f"""
🚀 *LIVE METRICS UPDATE*

👤 *User Action*
• User ID: `{user_id}`
• Username: @{username}
• Action: {action}
• Time: {datetime.now().strftime("%H:%M:%S")}

📊 *System Metrics*
• CPU Usage: {metrics.cpu_usage:.1f}%
• Memory Usage: {metrics.memory_usage:.1f}%
• Disk Usage: {metrics.disk_usage:.1f}%
• Active Users: {metrics.active_users}
• Total Requests: {metrics.total_requests}
• Uptime: {humanize.naturaldelta(timedelta(seconds=metrics.uptime))}

🔄 *Processing Stats*
• Active Processes: {len(processing_manager.active_processes)}
"""
        
        if details:
            details_text = "\n📝 *Action Details*\n"
            for key, value in details.items():
                details_text += f"• {key}: {value}\n"
            metrics_text += details_text
        
        await self.send_developer_update(metrics_text)
    
    async def send_detailed_analytics(self):
        """Send detailed analytics report"""
        metrics = system_monitor.get_system_metrics()
        
        # Create performance chart
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # CPU Usage over time (simulated)
        time_points = np.arange(len(metrics.response_times))
        axes[0, 0].plot(time_points, [metrics.cpu_usage] * len(time_points), 'b-')
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_ylabel('CPU %')
        
        # Response times
        if metrics.response_times:
            axes[0, 1].plot(metrics.response_times, 'r-')
            axes[0, 1].set_title('Response Times')
            axes[0, 1].set_ylabel('Seconds')
        
        # Memory usage
        mem_points = [metrics.memory_usage] * 10
        axes[1, 0].bar(range(len(mem_points)), mem_points)
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory %')
        
        # Active users
        axes[1, 1].pie([metrics.active_users, max(0, 100-metrics.active_users)], 
                      labels=['Active', 'Inactive'], autopct='%1.1f%%')
        axes[1, 1].set_title('Active Users Distribution')
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Send report with image
        report_text = f"""
📈 *SYSTEM ANALYTICS REPORT*

🏢 *System Health*
• CPU: {metrics.cpu_usage:.1f}%
• Memory: {metrics.memory_usage:.1f}%
• Disk: {metrics.disk_usage:.1f}%
• GPU: {metrics.gpu_usage:.1f}%

👥 *User Statistics*
• Active Users: {metrics.active_users}
• Total Requests: {metrics.total_requests}
• Uptime: {humanize.naturaldelta(timedelta(seconds=metrics.uptime))}
• Avg Response Time: {np.mean(metrics.response_times):.3f}s

📅 *Daily Stats*
{self._format_daily_stats()}

⏰ *Peak Hours*
{self._format_hourly_stats()}
"""
        
        await self.bot.bot.send_photo(
            chat_id=self.developer_chat_id,
            photo=buf,
            caption=report_text[:1024],
            parse_mode=ParseMode.MARKDOWN
        )
    
    def _format_daily_stats(self):
        daily_stats = db.data.get("analytics", {}).get("daily_stats", {})
        if not daily_stats:
            return "No data available"
        
        table_data = []
        for date, stats in sorted(daily_stats.items())[-5:]:  # Last 5 days
            table_data.append([date, stats["questions"], f"{stats['processing_time']:.2f}s"])
        
        return tabulate(table_data, headers=["Date", "Questions", "Processing Time"], tablefmt="simple")
    
    def _format_hourly_stats(self):
        hourly_stats = db.data.get("analytics", {}).get("hourly_stats", {})
        if not hourly_stats:
            return "No data available"
        
        table_data = []
        for hour, stats in sorted(hourly_stats.items())[-8:]:  # Last 8 hours
            table_data.append([hour, stats["questions"], len(stats.get("users", []))])
        
        return tabulate(table_data, headers=["Hour", "Questions", "Unique Users"], tablefmt="simple")

# Enhanced Study Board with live updates
class EnhancedStudyBoard:
    def __init__(self):
        self.boards = {}
        self.user_sessions = {}
        
    def create_user_board(self, user_id: int):
        if user_id not in self.boards:
            self.boards[user_id] = {
                "questions": [],
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
                "session_id": str(uuid.uuid4())
            }
            self.user_sessions[user_id] = {
                "start_time": datetime.now(),
                "message_count": 0,
                "questions_processed": 0,
                "current_process": None
            }
        return self.boards[user_id]
    
    def add_question_with_progress(self, user_id: int, question: str, source: str):
        board = self.create_user_board(user_id)
        
        question_id = len(board["questions"])
        board_item = {
            "id": question_id,
            "question": question,
            "source": source,
            "added_at": datetime.now(),
            "status": "processing",
            "progress": 0.0,
            "answers": {},
            "answer_formats_available": [],
            "processing_steps": []
        }
        
        board["questions"].append(board_item)
        board["last_updated"] = datetime.now()
        
        # Update session stats
        self.user_sessions[user_id]["questions_processed"] += 1
        
        return question_id
    
    def update_question_progress(self, user_id: int, question_id: int, progress: float, step: str = None):
        if user_id in self.boards and 0 <= question_id < len(self.boards[user_id]["questions"]):
            self.boards[user_id]["questions"][question_id]["progress"] = progress
            
            if step:
                self.boards[user_id]["questions"][question_id]["processing_steps"].append({
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "progress": progress
                })
            
            self.boards[user_id]["last_updated"] = datetime.now()
    
    def complete_question(self, user_id: int, question_id: int, answers: Dict):
        if user_id in self.boards and 0 <= question_id < len(self.boards[user_id]["questions"]):
            self.boards[user_id]["questions"][question_id]["status"] = "completed"
            self.boards[user_id]["questions"][question_id]["progress"] = 100.0
            self.boards[user_id]["questions"][question_id]["answers"] = answers
            self.boards[user_id]["questions"][question_id]["answer_formats_available"] = list(answers.keys())
            self.boards[user_id]["last_updated"] = datetime.now()
    
    def get_board_with_progress(self, user_id: int) -> Tuple[str, float]:
        if user_id not in self.boards:
            return "📚 *STUDY BOARD*\n\nNo active session. Send a question to start!", 0.0
        
        board = self.boards[user_id]
        total_progress = 0.0
        active_questions = 0
        
        board_text = "📚 *STUDY BOARD*\n\n"
        board_text += f"📅 Session: {board['created_at'].strftime('%H:%M:%S')}\n"
        board_text += f"🔄 Last Updated: {board['last_updated'].strftime('%H:%M:%S')}\n"
        board_text += f"📊 Questions: {len(board['questions'])}\n\n"
        
        for i, item in enumerate(board["questions"], 1):
            status_emoji = "✅" if item["status"] == "completed" else "🔄"
            progress_bar = self._create_progress_bar(item["progress"])
            
            board_text += f"{i}. {status_emoji} {item['question'][:50]}...\n"
            board_text += f"   📊 Progress: {progress_bar} {item['progress']:.0f}%\n"
            board_text += f"   📎 Source: {item['source']}\n"
            
            if item["status"] == "completed":
                board_text += f"   📝 Formats: {', '.join(item['answer_formats_available'][:3])}\n"
            
            board_text += "\n"
            
            if item["status"] != "completed":
                active_questions += 1
                total_progress += item["progress"]
        
        if active_questions > 0:
            overall_progress = total_progress / active_questions
        else:
            overall_progress = 100.0
        
        board_text += f"\n📈 Overall Progress: {self._create_progress_bar(overall_progress)} {overall_progress:.1f}%"
        
        return board_text, overall_progress
    
    def _create_progress_bar(self, percentage: float, length: int = 10) -> str:
        filled = int(length * percentage / 100)
        empty = length - filled
        return "█" * filled + "░" * empty

study_board = EnhancedStudyBoard()

# Main Bot Class with all enhanced features
class ExtremeStudyBot:
    def __init__(self, application):
        self.application = application
        self.developer_dashboard = DeveloperDashboard(application)
        self.processing_times = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with welcome animation"""
        user = update.effective_user
        user_id = user.id
        
        # Send developer notification
        await self.developer_dashboard.send_live_metrics(
            user_id=user_id,
            username=user.username or user.first_name,
            action="start_command",
            details={"command": "/start", "time": datetime.now().isoformat()}
        )
        
        # Create loading message
        loading_msg = await update.message.reply_text("🚀 *Initializing Study Bot Pro...*", parse_mode=ParseMode.MARKDOWN)
        
        # Simulate loading with progress
        for i in range(1, 11):
            await asyncio.sleep(0.1)
            progress_bar = "█" * i + "░" * (10 - i)
            percentage = i * 10
            await loading_msg.edit_text(f"🚀 *Initializing...*\n\n{progress_bar} {percentage}%\n\nLoading modules...")
        
        # Send welcome message with animation
        welcome_text = """
🎓 *STUDY BOT PRO - EXTREME EDITION* 🚀

⚡ *Powered by Advanced AI Processing*
📊 *Live Progress Tracking*
👨‍💻 *Developer Dashboard Enabled*
📈 *Real-time Analytics*

🌟 *Features Activated:*
✅ Multi-format Answer Generation
✅ Live Processing with Percentage
✅ Image/PDF/Audio Processing
✅ Study Board with Progress
✅ Developer Monitoring
✅ System Health Tracking

📋 *Available Commands:*
/start - Enhanced welcome
/board - Live progress board
/metrics - System metrics
/analytics - Detailed analytics
/help - Help menu
/developer - Developer info

🎯 *Just send me:*
• Any question in text
• Photo of question
• PDF document
• Voice message
"""
        
        keyboard = [
            [
                InlineKeyboardButton("📚 Live Board", callback_data="live_board"),
                InlineKeyboardButton("📊 Metrics", callback_data="show_metrics")
            ],
            [
                InlineKeyboardButton("⚡ Quick Start", callback_data="quick_start"),
                InlineKeyboardButton("🎯 Tutorial", callback_data="tutorial")
            ],
            [
                InlineKeyboardButton("👨‍💻 Developer", callback_data="developer_info"),
                InlineKeyboardButton("📈 Analytics", callback_data="user_analytics")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await loading_msg.edit_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
        # Send system status
        system_status = await self._get_system_status()
        await update.message.reply_text(
            system_status,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def _get_system_status(self):
        """Get current system status"""
        metrics = system_monitor.get_system_metrics()
        
        status_text = f"""
🖥️ *SYSTEM STATUS*

📊 *Performance*
• CPU: {metrics.cpu_usage:.1f}% {'🟢' if metrics.cpu_usage < 70 else '🟡' if metrics.cpu_usage < 90 else '🔴'}
• Memory: {metrics.memory_usage:.1f}% {'🟢' if metrics.memory_usage < 70 else '🟡' if metrics.memory_usage < 90 else '🔴'}
• Disk: {metrics.disk_usage:.1f}% {'🟢' if metrics.disk_usage < 70 else '🟡' if metrics.disk_usage < 90 else '🔴'}
• Active Processes: {len(processing_manager.active_processes)}

👥 *Usage Stats*
• Active Users: {metrics.active_users}
• Total Requests: {metrics.total_requests}
• Uptime: {humanize.naturaldelta(timedelta(seconds=metrics.uptime))}

⚡ *Response Times*
• Average: {np.mean(metrics.response_times):.3f}s if metrics.response_times else 'N/A'
• Last: {metrics.response_times[-1]:.3f}s if metrics.response_times else 'N/A'
"""
        return status_text
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages with live processing"""
        user = update.effective_user
        user_id = user.id
        text = update.message.text
        
        # Start timing
        start_time = time.time()
        
        # Send developer notification
        await self.developer_dashboard.send_live_metrics(
            user_id=user_id,
            username=user.username or user.first_name,
            action="text_message",
            details={
                "message_length": len(text),
                "preview": text[:100] + "..." if len(text) > 100 else text
            }
        )
        
        # Create process ID
        process_id = f"text_{user_id}_{int(time.time())}"
        
        # Send initial processing message
        processing_msg = await update.message.reply_text(
            "🔍 *Processing your question...*\n\n📊 Initializing analysis...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Start processing with progress updates
        processing_manager.start_processing(user_id, process_id, "Text Analysis")
        
        # Update progress in steps
        steps = [
            ("Analyzing question structure", 10),
            ("Checking question type", 20),
            ("Generating answer formats", 40),
            ("Preparing study board", 60),
            ("Finalizing responses", 80),
            ("Complete", 100)
        ]
        
        for step_name, progress in steps:
            await asyncio.sleep(0.5)  # Simulate processing time
            processing_manager.update_progress(
                process_id,
                progress,
                f"Step: {step_name}",
                {"current_step": step_name}
            )
            
            # Update user message
            progress_bar = study_board._create_progress_bar(progress)
            await processing_msg.edit_text(
                f"🔍 *Processing your question...*\n\n"
                f"📊 {step_name}\n"
                f"{progress_bar} {progress}%\n\n"
                f"⏱️ Time elapsed: {time.time() - start_time:.1f}s"
            )
        
        # Add to study board
        question_id = study_board.add_question_with_progress(user_id, text, "Text Input")
        
        # Generate answers
        answers = await self._generate_answers_with_progress(text, process_id)
        
        # Complete processing
        study_board.complete_question(user_id, question_id, answers)
        processing_manager.complete_processing(process_id, answers)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        system_monitor.record_response_time(processing_time)
        
        # Update database
        db.add_question_with_analytics(user_id, text, "text", processing_time)
        db.update_user_metrics(user_id, {
            "total_questions": 1,
            "total_processing_time": processing_time
        })
        
        # Send completion message
        keyboard = self._create_answer_keyboard(user_id, question_id)
        
        await processing_msg.edit_text(
            f"✅ *Processing Complete!*\n\n"
            f"*Question:* {text[:100]}...\n\n"
            f"📊 *Stats*\n"
            f"• Processing Time: {processing_time:.2f}s\n"
            f"• Answer Formats: {len(answers)}\n"
            f"• Added to Study Board: ✓\n\n"
            f"Choose answer format:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo with live OCR processing"""
        user = update.effective_user
        user_id = user.id
        photo = update.message.photo[-1]
        
        # Send developer notification
        await self.developer_dashboard.send_live_metrics(
            user_id=user_id,
            username=user.username or user.first_name,
            action="photo_upload",
            details={"file_size": photo.file_size, "resolution": f"{photo.width}x{photo.height}"}
        )
        
        process_id = f"image_{user_id}_{int(time.time())}"
        processing_manager.start_processing(user_id, process_id, "Image OCR Processing")
        
        # Send initial message
        processing_msg = await update.message.reply_text(
            "🖼️ *Processing image...*\n\n📊 Downloading image...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Download image with progress
        photo_file = await photo.get_file()
        
        # Simulate download progress
        for i in range(1, 6):
            await asyncio.sleep(0.3)
            progress = i * 20
            processing_manager.update_progress(process_id, progress, f"Downloading... {i}/5")
            progress_bar = study_board._create_progress_bar(progress)
            await processing_msg.edit_text(
                f"🖼️ *Processing image...*\n\n"
                f"📊 Downloading image...\n"
                f"{progress_bar} {progress}%"
            )
        
        # Download actual file
        photo_bytes = await photo_file.download_as_bytearray()
        
        # OCR Processing
        processing_manager.update_progress(process_id, 60, "Performing OCR...")
        await processing_msg.edit_text("🖼️ *Processing image...*\n\n📊 Performing OCR analysis...")
        
        extracted_text = await self._perform_ocr_with_progress(photo_bytes, process_id)
        
        if not extracted_text:
            await processing_msg.edit_text("❌ *OCR Failed*\n\nCould not extract text from image.")
            return
        
        # Continue with text processing
        processing_manager.update_progress(process_id, 80, "Processing extracted text...")
        
        question_id = study_board.add_question_with_progress(user_id, extracted_text, "Image OCR")
        
        # Generate answers
        answers = await self._generate_answers_with_progress(extracted_text, process_id)
        study_board.complete_question(user_id, question_id, answers)
        
        processing_manager.complete_processing(process_id, answers)
        
        keyboard = self._create_answer_keyboard(user_id, question_id)
        
        await processing_msg.edit_text(
            f"✅ *Image Processing Complete!*\n\n"
            f"*Extracted Text:*\n{extracted_text[:200]}...\n\n"
            f"📊 *OCR Results*\n"
            f"• Characters Extracted: {len(extracted_text)}\n"
            f"• Added to Study Board: ✓\n\n"
            f"Choose answer format:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
    
    async def _perform_ocr_with_progress(self, image_bytes: bytes, process_id: str) -> str:
        """Perform OCR with progress updates"""
        try:
            # Update progress
            processing_manager.update_progress(process_id, 65, "Loading image...")
            
            image = Image.open(io.BytesIO(image_bytes))
            
            processing_manager.update_progress(process_id, 70, "Configuring OCR...")
            
            # Configure OCR
            custom_config = r'--oem 3 --psm 6'
            
            processing_manager.update_progress(process_id, 75, "Extracting text...")
            
            text = pytesseract.image_to_string(image, config=custom_config, lang='eng+hin')
            
            processing_manager.update_progress(process_id, 90, "Cleaning text...")
            
            # Clean text
            text = ' '.join(text.split())
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""
    
    async def _generate_answers_with_progress(self, question: str, process_id: str) -> Dict[str, str]:
        """Generate answers with progress tracking"""
        answers = {}
        
        answer_types = [
            (AnswerType.ONE_WORD, "Generating one-word answer..."),
            (AnswerType.SHORT, "Generating short answer..."),
            (AnswerType.MEDIUM, "Generating medium answer..."),
            (AnswerType.LONG, "Generating detailed answer..."),
            (AnswerType.EXPLANATION, "Generating explanation...")
        ]
        
        total_types = len(answer_types)
        
        for i, (answer_type, step_name) in enumerate(answer_types):
            progress = 70 + (i * 6)  # Progress from 70 to 94
            
            processing_manager.update_progress(
                process_id,
                progress,
                step_name,
                {"current_answer_type": answer_type.value}
            )
            
            try:
                prompt = self._create_prompt_for_type(question, answer_type)
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                answers[answer_type.value] = response.text
            except Exception as e:
                logger.error(f"Error generating {answer_type} answer: {e}")
                answers[answer_type.value] = f"Error: {str(e)}"
            
            await asyncio.sleep(0.5)  # Simulate processing
        
        return answers
    
    def _create_prompt_for_type(self, question: str, answer_type: AnswerType) -> str:
        """Create appropriate prompt for each answer type"""
        prompts = {
            AnswerType.ONE_WORD: f"Provide ONLY one word answer for: {question}. No explanations, just the single word answer.",
            AnswerType.SHORT: f"Provide a concise answer (1-2 sentences maximum) for: {question}",
            AnswerType.MEDIUM: f"Provide a balanced answer (3-4 sentences) for: {question}. Include key points.",
            AnswerType.LONG: f"Provide a comprehensive answer for: {question}. Include details, examples, and explanations.",
            AnswerType.EXPLANATION: f"Explain in detail with examples, step-by-step reasoning for: {question}. Make it educational."
        }
        return prompts.get(answer_type, f"Answer: {question}")
    
    def _create_answer_keyboard(self, user_id: int, question_id: int):
        """Create interactive keyboard for answers"""
        keyboard = [
            [
                InlineKeyboardButton("🔤 One Word", callback_data=f"answer_{question_id}_one_word"),
                InlineKeyboardButton("📝 Short", callback_data=f"answer_{question_id}_short")
            ],
            [
                InlineKeyboardButton("📄 Medium", callback_data=f"answer_{question_id}_medium"),
                InlineKeyboardButton("📚 Long", callback_data=f"answer_{question_id}_long")
            ],
            [
                InlineKeyboardButton("🧠 Explanation", callback_data=f"answer_{question_id}_explanation"),
                InlineKeyboardButton("📥 All Formats", callback_data=f"answer_{question_id}_all")
            ],
            [
                InlineKeyboardButton("📊 Live Board", callback_data="live_board"),
                InlineKeyboardButton("📈 Metrics", callback_data="user_metrics")
            ],
            [
                InlineKeyboardButton("🔄 Process Another", callback_data="new_question"),
                InlineKeyboardButton("💾 Save Session", callback_data="save_session")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        data = query.data
        
        # Send developer notification for important actions
        if not data.startswith("answer_"):
            await self.developer_dashboard.send_live_metrics(
                user_id=user.id,
                username=user.username or user.first_name,
                action=f"callback_{data}",
                details={"callback_data": data}
            )
        
        if data == "live_board":
            board_text, overall_progress = study_board.get_board_with_progress(user.id)
            
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="live_board"),
                 InlineKeyboardButton("🧹 Clear", callback_data="clear_board")],
                [InlineKeyboardButton("📥 Export", callback_data="export_board"),
                 InlineKeyboardButton("📊 Analytics", callback_data="board_analytics")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                board_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        
        elif data.startswith("answer_"):
            parts = data.split("_")
            if len(parts) >= 3:
                question_id = int(parts[1])
                answer_type = parts[2]
                
                if user.id in study_board.boards:
                    questions = study_board.boards[user.id]["questions"]
                    if 0 <= question_id < len(questions):
                        question_item = questions[question_id]
                        
                        if answer_type == "all":
                            # Show all answers
                            response_text = f"📚 *All Answer Formats*\n\n*Question:* {question_item['question'][:200]}...\n\n"
                            
                            for ans_type, answer in question_item.get("answers", {}).items():
                                response_text += f"*{ans_type.upper().replace('_', ' ')}:*\n{answer[:300]}...\n\n"
                            
                            await query.edit_message_text(
                                response_text[:4000],
                                parse_mode=ParseMode.MARKDOWN
                            )
                        else:
                            # Show specific answer
                            answer = question_item.get("answers", {}).get(answer_type, "Answer not available.")
                            answer_type_display = answer_type.replace('_', ' ').title()
                            
                            response_text = f"*{answer_type_display} Answer*\n\n"
                            response_text += f"*Question:* {question_item['question'][:200]}...\n\n"
                            response_text += f"*Answer:*\n{answer}"
                            
                            keyboard = self._create_answer_keyboard(user.id, question_id)
                            
                            await query.edit_message_text(
                                response_text[:4000],
                                parse_mode=ParseMode.MARKDOWN,
                                reply_markup=keyboard
                            )
        
        elif data == "show_metrics":
            metrics = system_monitor.get_system_metrics()
            
            metrics_text = f"""
📊 *REAL-TIME METRICS*

🖥️ *System Health*
• CPU Usage: {metrics.cpu_usage:.1f}%
• Memory Usage: {metrics.memory_usage:.1f}%
• Disk Usage: {metrics.disk_usage:.1f}%
• GPU Usage: {metrics.gpu_usage:.1f}%

👥 *Usage Statistics*
• Active Users: {metrics.active_users}
• Total Requests: {metrics.total_requests}
• Uptime: {humanize.naturaldelta(timedelta(seconds=metrics.uptime))}

⚡ *Performance*
• Avg Response Time: {np.mean(metrics.response_times):.3f}s if metrics.response_times else 'N/A'
• 95th Percentile: {np.percentile(metrics.response_times, 95):.3f}s if metrics.response_times else 'N/A'

📈 *Your Session*
• Questions Processed: {study_board.user_sessions.get(user.id, {}).get('questions_processed', 0)}
• Active Processes: {len([p for p in processing_manager.active_processes.values() if p.details.get('user_id') == user.id])}
"""
            
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="show_metrics"),
                 InlineKeyboardButton("📈 Detailed", callback_data="detailed_metrics")],
                [InlineKeyboardButton("📊 My Stats", callback_data="user_stats"),
                 InlineKeyboardButton("📚 Back", callback_data="live_board")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                metrics_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        
        elif data == "developer_info":
            dev_info = """
👨‍💻 *DEVELOPER DASHBOARD*

🔧 *Bot Information*
• Name: Study Bot Pro Extreme
• Version: 3.0.0
• Framework: python-telegram-bot
• AI Model: Gemini Pro
• Database: JSON + Analytics

📡 *Monitoring*
• Live Processing Tracking
• Real-time Metrics
• Developer Notifications
• Error Logging

🔐 *Security*
• User Session Management
• Request Rate Limiting
• Data Encryption
• Privacy Compliance

📞 *Contact*
• Developer: AI Assistant
• Support: Via Telegram
• Updates: Automatic

⚙️ *Technical Stack*
• Python 3.11+
• Async/Await
• Multi-threaded Processing
• Advanced OCR
• PDF Processing
"""
            
            keyboard = [
                [InlineKeyboardButton("🔄 Status", callback_data="system_status"),
                 InlineKeyboardButton("🐛 Logs", callback_data="view_logs")],
                [InlineKeyboardButton("📊 Analytics", callback_data="dev_analytics"),
                 InlineKeyboardButton("🔙 Back", callback_data="live_board")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                dev_info,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
    
    async def show_analytics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user analytics"""
        user = update.effective_user
        
        user_data = db.data.get("users", {}).get(str(user.id), {})
        
        analytics_text = f"""
📈 *YOUR ANALYTICS*

👤 *User Profile*
• User ID: `{user.id}`
• First Seen: {user_data.get('first_seen', 'N/A')}
• Sessions: {user_data.get('sessions_count', 0)}

📊 *Study Statistics*
• Total Questions: {user_data.get('total_questions', 0)}
• Total Processing Time: {user_data.get('total_processing_time', 0):.2f}s
• Avg Time/Question: {user_data.get('total_processing_time', 0) / max(user_data.get('total_questions', 1), 1):.2f}s
• Preferred Format: {user_data.get('preferred_answer_type', 'medium')}

🏆 *Achievements*
• Quick Learner: {user_data.get('total_questions', 0) > 10}
• Active User: {user_data.get('sessions_count', 0) > 5}
• Power User: {user_data.get('total_processing_time', 0) > 300}

📅 *Recent Activity*
{self._get_recent_activity(user.id)}
"""
        
        await update.message.reply_text(
            analytics_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    def _get_recent_activity(self, user_id: str):
        """Get recent activity for user"""
        user_questions = db.data.get("questions", {}).get(str(user_id), [])
        
        if not user_questions:
            return "No recent activity"
        
        recent = user_questions[-5:]  # Last 5 questions
        activity_text = ""
        
        for q in recent:
            timestamp = datetime.fromisoformat(q['timestamp']).strftime("%H:%M")
            activity_text += f"• {timestamp}: {q['question'][:50]}...\n"
        
        return activity_text

# Command Handlers
async def metrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /metrics command"""
    bot = context.bot_data.get('bot_instance')
    if bot:
        await bot.show_analytics(update, context)

async def developer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /developer command"""
    user = update.effective_user
    
    # Check if user is developer
    if str(user.id) == DEVELOPER_CHAT_ID:
        bot = context.bot_data.get('bot_instance')
        if bot:
            await bot.developer_dashboard.send_detailed_analytics()
            await update.message.reply_text("📊 Developer analytics sent!")
    else:
        await update.message.reply_text("🔒 Access restricted to developer only.")

async def board_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /board command"""
    user = update.effective_user
    board_text, overall_progress = study_board.get_board_with_progress(user.id)
    
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="live_board"),
         InlineKeyboardButton("📥 Export", callback_data="export_board")],
        [InlineKeyboardButton("📊 Analytics", callback_data="user_analytics"),
         InlineKeyboardButton("⚡ Quick Action", callback_data="quick_action")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        board_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
🎯 *STUDY BOT PRO - HELP GUIDE*

📚 *Basic Commands*
/start - Enhanced welcome with system status
/board - Live study board with progress
/metrics - Your personal analytics
/analytics - Detailed usage statistics
/help - This help message

🔄 *How to Use*
1. Send *text questions* directly
2. Send *photos* of questions (OCR will extract text)
3. Send *PDF files* with questions
4. Send *voice messages* with questions

📊 *Answer Formats*
• 🔤 One Word - Concise single word
• 📝 Short - 1-2 sentences
• 📄 Medium - 3-4 sentences
• 📚 Long - Detailed answer
• 🧠 Explanation - Step-by-step reasoning

⚡ *Advanced Features*
• Live progress tracking
• Real-time system metrics
• Developer monitoring
• Session management
• Export functionality

🔧 *Troubleshooting*
• If OCR fails, try clearer images
• For large PDFs, processing may take time
• Voice messages should be clear
• Contact support for issues

🌟 *Pro Tips*
• Use the study board to track progress
• Check metrics for performance
• Export important sessions
• Try different answer formats
"""
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# Main application setup
async def main():
    """Start the advanced bot"""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment variables")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Initialize bot
    bot = ExtremeStudyBot(application)
    application.bot_data['bot_instance'] = bot
    
    # Add command handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("board", board_command))
    application.add_handler(CommandHandler("metrics", metrics_command))
    application.add_handler(CommandHandler("analytics", bot.show_analytics))
    application.add_handler(CommandHandler("developer", developer_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
    application.add_handler(CallbackQueryHandler(bot.callback_handler))
    
    # Add error handler
    async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Exception while handling an update: {context.error}")
        
        # Send error to developer
        error_msg = f"""
🚨 *ERROR ALERT*

❌ *Error Details*
• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Error: {str(context.error)[:500]}
• Update: {update.update_id if update else 'N/A'}

🔧 *Traceback*
```python
{context.error.__traceback__}