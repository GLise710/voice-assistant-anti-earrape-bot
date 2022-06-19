import asyncio
import io
from typing import Dict

import discord
import functools
import numpy as np
import speech_recognition as sr
from discord.ext import commands
from scipy.io.wavfile import write
import youtube_dl

import math
import itertools
import timeout
import random

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix=">", intents=intents)

connections = {}

import threading
import time
import wave as wav

DISCORD_TOKEN = "OTg2NzE1MjY2MDUyNDAzMjMw.GnUKGD.9RnZnNhvtJ8eDcs-9hwJPl1iTip_tC85EIsLLI"


message_sender_queue = []
def send_message(channel, message):
    message_sender_queue.append((channel, message))

muter_queue = []
def mute_user(guild, user, mute_state, after=-1):
    muter_queue.append((guild, user, mute_state, time.time() + after))

async def message_sender_function():
    while True:
        if len(message_sender_queue) == 0:
            await asyncio.sleep(0.001)
            continue
        
        channel, message = message_sender_queue.pop(0)
        await channel.send(message)

async def muter_function():
    while True:
        for index, mute_request in enumerate(muter_queue):
            guild, user, mute_state, when = mute_request
            if time.time() > when:
                user_obj = await guild.fetch_member(user)
                await user_obj.edit(mute=mute_state)
                del muter_queue[index]
        
        await asyncio.sleep(0.001)

class AntiEarrape:
    def __init__(self, guild):
        self.volume_history = {}
        self.muted_users = {}

        self.guild = guild
    
    def new_packet(self, data, user):
        if user not in self.volume_history:
            self.volume_history[user] = []

        audio_array = np.frombuffer(data, dtype=np.int32)
        audio_array = np.ndarray.astype(audio_array, dtype=np.int16)
        audio_array = np.ndarray.astype(audio_array, dtype=np.float32)

        if len(audio_array) == 0:
            return

        rms = audio_array * audio_array
        rms = rms[np.argsort(rms)[-100:]]

        current_volume = np.mean(rms)
        current_volume = current_volume**0.5
        self.volume_history[user].append(current_volume)
        if len(self.volume_history[user]) > 10000:
            self.volume_history[user].pop(0)

        if len(self.volume_history[user]) < 200:
            return
        
        # print(current_volume, np.mean(self.volume_history[user]) * 5)

        latest = np.mean(self.volume_history[user][-5:])

        if latest > np.mean(self.volume_history[user]) * 6:
            if user not in self.muted_users:
                self.muted_users[user] = 0
            
            if time.time() > self.muted_users[user] + 3:
                print(f"{user} is earraping")
                self.muted_users[user] = time.time()
                mute_user(self.guild, user, True)
                mute_user(self.guild, user, False, 3)


# Silence useless bug reports messages
youtube_dl.utils.bug_reports_message = lambda: ''


class VoiceError(Exception):
    pass


class YTDLError(Exception):
    pass


class YTDLSource(discord.PCMVolumeTransformer):
    YTDL_OPTIONS = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
        'restrictfilenames': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'logtostderr': False,
        'quiet': True,
        'no_warnings': True,
        'default_search': 'auto',
        'source_address': '0.0.0.0',
    }

    FFMPEG_OPTIONS = {
        'before_options': '',
        'options': '-vn',
    }

    ytdl = youtube_dl.YoutubeDL(YTDL_OPTIONS)

    def __init__(self, ctx: commands.Context, source: discord.FFmpegPCMAudio, *, data: dict, volume: float = 0.5):
        super().__init__(source, volume)

        self.requester = ctx.author
        self.channel = ctx.channel
        self.data = data

        self.uploader = data.get('uploader')
        self.uploader_url = data.get('uploader_url')
        date = data.get('upload_date')
        self.upload_date = date[6:8] + '.' + date[4:6] + '.' + date[0:4]
        self.title = data.get('title')
        self.thumbnail = data.get('thumbnail')
        self.description = data.get('description')
        self.duration = self.parse_duration(int(data.get('duration')))
        self.tags = data.get('tags')
        self.url = data.get('webpage_url')
        self.views = data.get('view_count')
        self.likes = data.get('like_count')
        self.dislikes = data.get('dislike_count')
        self.stream_url = data.get('url')

    def __str__(self):
        return '**{0.title}** by **{0.uploader}**'.format(self)

    @classmethod
    async def create_source(cls, ctx: commands.Context, search: str, *, loop: asyncio.BaseEventLoop = None):
        loop = loop or asyncio.get_event_loop()

        partial = functools.partial(cls.ytdl.extract_info, search, download=False, process=False)
        data = await loop.run_in_executor(None, partial)

        if data is None:
            raise YTDLError('Couldn\'t find anything that matches `{}`'.format(search))

        if 'entries' not in data:
            process_info = data
        else:
            process_info = None
            for entry in data['entries']:
                if entry:
                    process_info = entry
                    break

            if process_info is None:
                raise YTDLError('Couldn\'t find anything that matches `{}`'.format(search))

        webpage_url = process_info['webpage_url']
        partial = functools.partial(cls.ytdl.extract_info, webpage_url, download=False)
        processed_info = await loop.run_in_executor(None, partial)

        if processed_info is None:
            raise YTDLError('Couldn\'t fetch `{}`'.format(webpage_url))

        if 'entries' not in processed_info:
            info = processed_info
        else:
            info = None
            while info is None:
                try:
                    info = processed_info['entries'].pop(0)
                except IndexError:
                    raise YTDLError('Couldn\'t retrieve any matches for `{}`'.format(webpage_url))

        return cls(ctx, discord.FFmpegPCMAudio("sample.mp3", **cls.FFMPEG_OPTIONS), data=info)

    @staticmethod
    def parse_duration(duration: int):
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        duration = []
        if days > 0:
            duration.append('{} days'.format(days))
        if hours > 0:
            duration.append('{} hours'.format(hours))
        if minutes > 0:
            duration.append('{} minutes'.format(minutes))
        if seconds > 0:
            duration.append('{} seconds'.format(seconds))

        return ', '.join(duration)



# @bot.command(name='play_song', help='To play song')
# async def play(ctx,url):
    
#     if not ctx.message.author.name=="Rohan Krishna" :
#          await ctx.send('NOT AUTHORISED!')
#          return
#     try :
#         server = ctx.message.guild
#         voice_channel = server.voice_client

#         async with ctx.typing():
#             filename = await YTDLSource.from_url(url, loop=bot.loop)
#             voice_channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=filename))
#         await ctx.send('**Now playing:** {}'.format(filename))
#     except:
#         await ctx.send("The bot is not connected to a voice channel.")



class Song:
    __slots__ = ('source', 'requester')

    def __init__(self, source: YTDLSource):
        self.source = source
        self.requester = source.requester

    def create_embed(self):
        embed = (discord.Embed(title='Now playing',
                               description='```css\n{0.source.title}\n```'.format(self),
                               color=discord.Color.blurple())
                 .add_field(name='Duration', value=self.source.duration)
                 .add_field(name='Requested by', value=self.requester.mention)
                 .add_field(name='Uploader', value='[{0.source.uploader}]({0.source.uploader_url})'.format(self))
                 .add_field(name='URL', value='[Click]({0.source.url})'.format(self))
                 .set_thumbnail(url=self.source.thumbnail))

        return embed


class SongQueue(asyncio.Queue):
    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(itertools.islice(self._queue, item.start, item.stop, item.step))
        else:
            return self._queue[item]

    def __iter__(self):
        return self._queue.__iter__()

    def __len__(self):
        return self.qsize()

    def clear(self):
        self._queue.clear()

    def shuffle(self):
        random.shuffle(self._queue)

    def remove(self, index: int):
        del self._queue[index]


class VoiceState:
    def __init__(self, bot: commands.Bot, ctx: commands.Context):
        self.bot = bot
        self._ctx = ctx

        self.current = None
        self.voice = None
        self.next = asyncio.Event()
        self.songs = SongQueue()

        self._loop = False
        self._volume = 0.5
        self.skip_votes = set()

        self.audio_player = bot.loop.create_task(self.audio_player_task())

    def __del__(self):
        self.audio_player.cancel()

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, value: bool):
        self._loop = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = value

    @property
    def is_playing(self):
        return self.voice and self.current

    async def audio_player_task(self):
        while True:
            self.next.clear()

            if not self.loop:
                # Try to get the next song within 3 minutes.
                # If no song will be added to the queue in time,
                # the player will disconnect due to performance
                # reasons.
                try:
                    async with timeout(180):  # 3 minutes
                        self.current = await self.songs.get()
                except asyncio.TimeoutError:
                    self.bot.loop.create_task(self.stop())
                    return

            self.current.source.volume = self._volume
            self.voice.play(self.current.source, after=self.play_next_song)
            await self.current.source.channel.send(embed=self.current.create_embed())

            await self.next.wait()

    def play_next_song(self, error=None):
        print("next song")
        if error:
            print(error)
            raise VoiceError(str(error))
            

        self.next.set()

    def skip(self):
        self.skip_votes.clear()

        if self.is_playing:
            self.voice.stop()

    async def stop(self):
        self.songs.clear()

        if self.voice:
            await self.voice.disconnect()
            self.voice = None


class Music(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.voice_states = {}

    def get_voice_state(self, ctx: commands.Context):
        state = self.voice_states.get(ctx.guild.id)
        if not state:
            state = VoiceState(self.bot, ctx)
            self.voice_states[ctx.guild.id] = state

        return state

    def cog_unload(self):
        for state in self.voice_states.values():
            self.bot.loop.create_task(state.stop())

    def cog_check(self, ctx: commands.Context):
        if not ctx.guild:
            raise commands.NoPrivateMessage('This command can\'t be used in DM channels.')

        return True

    async def cog_before_invoke(self, ctx: commands.Context):
        ctx.voice_state = self.get_voice_state(ctx)

    async def cog_command_error(self, ctx: commands.Context, error: commands.CommandError):
        await ctx.send('An error occurred: {}'.format(str(error)))

    @commands.command(name='join', invoke_without_subcommand=True)
    async def _join(self, ctx: commands.Context):
        """Joins a voice channel."""

        destination = ctx.author.voice.channel
        if ctx.voice_state.voice:
            await ctx.voice_state.voice.move_to(destination)
            return

        ctx.voice_state.voice = await destination.connect()

    @commands.command(name='summon')
    @commands.has_permissions(manage_guild=True)
    async def _summon(self, ctx: commands.Context, *, channel: discord.VoiceChannel = None):
        """Summons the bot to a voice channel.
        If no channel was specified, it joins your channel.
        """

        if not channel and not ctx.author.voice:
            raise VoiceError('You are neither connected to a voice channel nor specified a channel to join.')

        destination = channel or ctx.author.voice.channel
        if ctx.voice_state.voice:
            await ctx.voice_state.voice.move_to(destination)
            return

        ctx.voice_state.voice = await destination.connect()

    @commands.command(name='leave', aliases=['disconnect'])
    @commands.has_permissions(manage_guild=True)
    async def _leave(self, ctx: commands.Context):
        """Clears the queue and leaves the voice channel."""

        if not ctx.voice_state.voice:
            return await ctx.send('Not connected to any voice channel.')

        await ctx.voice_state.stop()
        del self.voice_states[ctx.guild.id]

    @commands.command(name='volume')
    async def _volume(self, ctx: commands.Context, *, volume: int):
        """Sets the volume of the player."""

        if not ctx.voice_state.is_playing:
            return await ctx.send('Nothing being played at the moment.')

        if 0 > volume > 100:
            return await ctx.send('Volume must be between 0 and 100')

        ctx.voice_state.volume = volume / 100
        await ctx.send('Volume of the player set to {}%'.format(volume))

    @commands.command(name='now', aliases=['current', 'playing'])
    async def _now(self, ctx: commands.Context):
        """Displays the currently playing song."""

        await ctx.send(embed=ctx.voice_state.current.create_embed())

    @commands.command(name='pause')
    @commands.has_permissions(manage_guild=True)
    async def _pause(self, ctx: commands.Context):
        """Pauses the currently playing song."""

        if not ctx.voice_state.is_playing and ctx.voice_state.voice.is_playing():
            ctx.voice_state.voice.pause()
            await ctx.message.add_reaction('⏯')

    @commands.command(name='resume')
    @commands.has_permissions(manage_guild=True)
    async def _resume(self, ctx: commands.Context):
        """Resumes a currently paused song."""

        if not ctx.voice_state.is_playing and ctx.voice_state.voice.is_paused():
            ctx.voice_state.voice.resume()
            await ctx.message.add_reaction('⏯')

    @commands.command(name='stop')
    @commands.has_permissions(manage_guild=True)
    async def _stop(self, ctx: commands.Context):
        """Stops playing song and clears the queue."""

        ctx.voice_state.songs.clear()

        if not ctx.voice_state.is_playing:
            ctx.voice_state.voice.stop()
            await ctx.message.add_reaction('⏹')

    @commands.command(name='skip')
    async def _skip(self, ctx: commands.Context):
        """Vote to skip a song. The requester can automatically skip.
        3 skip votes are needed for the song to be skipped.
        """

        if not ctx.voice_state.is_playing:
            return await ctx.send('Not playing any music right now...')

        voter = ctx.message.author
        if voter == ctx.voice_state.current.requester:
            await ctx.message.add_reaction('⏭')
            ctx.voice_state.skip()

        elif voter.id not in ctx.voice_state.skip_votes:
            ctx.voice_state.skip_votes.add(voter.id)
            total_votes = len(ctx.voice_state.skip_votes)

            if total_votes >= 3:
                await ctx.message.add_reaction('⏭')
                ctx.voice_state.skip()
            else:
                await ctx.send('Skip vote added, currently at **{}/3**'.format(total_votes))

        else:
            await ctx.send('You have already voted to skip this song.')

    @commands.command(name='queue')
    async def _queue(self, ctx: commands.Context, *, page: int = 1):
        """Shows the player's queue.
        You can optionally specify the page to show. Each page contains 10 elements.
        """

        if len(ctx.voice_state.songs) == 0:
            return await ctx.send('Empty queue.')

        items_per_page = 10
        pages = math.ceil(len(ctx.voice_state.songs) / items_per_page)

        start = (page - 1) * items_per_page
        end = start + items_per_page

        queue = ''
        for i, song in enumerate(ctx.voice_state.songs[start:end], start=start):
            queue += '`{0}.` [**{1.source.title}**]({1.source.url})\n'.format(i + 1, song)

        embed = (discord.Embed(description='**{} tracks:**\n\n{}'.format(len(ctx.voice_state.songs), queue))
                 .set_footer(text='Viewing page {}/{}'.format(page, pages)))
        await ctx.send(embed=embed)

    @commands.command(name='shuffle')
    async def _shuffle(self, ctx: commands.Context):
        """Shuffles the queue."""

        if len(ctx.voice_state.songs) == 0:
            return await ctx.send('Empty queue.')

        ctx.voice_state.songs.shuffle()
        await ctx.message.add_reaction('✅')

    @commands.command(name='remove')
    async def _remove(self, ctx: commands.Context, index: int):
        """Removes a song from the queue at a given index."""

        if len(ctx.voice_state.songs) == 0:
            return await ctx.send('Empty queue.')

        ctx.voice_state.songs.remove(index - 1)
        await ctx.message.add_reaction('✅')

    @commands.command(name='loop')
    async def _loop(self, ctx: commands.Context):
        """Loops the currently playing song.
        Invoke this command again to unloop the song.
        """

        if not ctx.voice_state.is_playing:
            return await ctx.send('Nothing being played at the moment.')

        # Inverse boolean value to loop and unloop.
        ctx.voice_state.loop = not ctx.voice_state.loop
        await ctx.message.add_reaction('✅')

    @commands.command(name='play')
    async def _play(self, ctx: commands.Context, *, search: str):
        """Plays a song.
        If there are songs in the queue, this will be queued until the
        other songs finished playing.
        This command automatically searches from various sites if no URL is provided.
        A list of these sites can be found here: https://rg3.github.io/youtube-dl/supportedsites.html
        """

        if not ctx.voice_state.voice:
            await ctx.invoke(self._join)

        async with ctx.typing():
            try:
                source = await YTDLSource.create_source(ctx, search, loop=self.bot.loop)
            except YTDLError as e:
                await ctx.send('An error occurred while processing this request: {}'.format(str(e)))
            else:
                song = Song(source)

                await ctx.voice_state.songs.put(song)
                await ctx.send('Enqueued {}'.format(str(source)))


class AudioHistory:
    def __init__(self):
        self.history = {}
        self.user_queue = {}
        # self.file = wav.open('sound.wav','wb')
        # self.file.setnchannels(2)
        # self.file.setsampwidth(2)
        # self.file.setframerate(48000)

        self.allow_delete: Dict[str, bool] = {}
        self.sphinx_counter = 0

    def new_packet(self, data, user):
        if user not in self.history:
            self.history[user] = []
            self.allow_delete[user] = True
            self.user_queue[user] = []
            threading.Thread(target=self.thread_loop, args=(user,)).start()

        self.user_queue[user].append(data)

    def thread_loop(self, user):
        packet_counter = 0
        last_packet_time = time.time()

        while True:
            if len(self.user_queue[user]) == 0:
                time.sleep(0.01)
                continue

            new_packet = self.user_queue[user].pop(0)

            if len(self.history[user]) > 400:
                if self.allow_delete[user]:
                    self.history[user].pop(0)

            self.history[user].append(new_packet)

            packet_counter += 1

            if packet_counter % 100 != 0:
                continue

            KEYWORD = "hello"

            text = self.sphinx(b''.join(self.history[user]))
            if KEYWORD in text:
                self.allow_delete[user] = False
                location = text.find(KEYWORD)
                text = text[location + len(KEYWORD) + 1:]
                print("hey robot detected, text is", text)

                if time.time() - last_packet_time > 2 and text != "":
                    words = text.split()

                    if words[0] == "play":
                        channel = bot.guilds[0].text_channels[0]
                        message = ";;play " + " ".join(words[1:])

                        send_message(channel, message)
                    
                    # reset
                    self.allow_delete[user] = True
                    self.history[user] = []

            last_packet_time = time.time()

    def sphinx(self, audio):
        byte_io = io.BytesIO()

        # convert to numpy array
        audio_array = np.frombuffer(audio, dtype=np.int32)
        audio_array = np.ndarray.astype(audio_array, dtype=np.int16)
        write(byte_io, 48000, audio_array)
        result_bytes = byte_io.read()

        with open("sound.wav", "wb") as f:
            f.write(result_bytes)

        audio_data = sr.AudioData(result_bytes, 48000, 2)
        r = sr.Recognizer()

        try:
            text = r.recognize_google(audio_data).lower()
            print("Google thinks you said " + text)

        except sr.UnknownValueError:
            text = "Google could not understand audio, try again"
            print("Google could not understand audio, try again")

        except sr.RequestError as e:
            print("Google error; {0}".format(e))
            text = "Google error; {0}".format(e)

        return text
        # spoj, feedni Googlu
        # keď započuješ začiatok commandu, prestaň mazať zo začiatku kým čakáš do zvyšku

    def process_sphinx_result(self, result: str):
        """
        :result: return value of self.sphinx()
        """

        if "play" in result:
            # words = result.split()
            send_message(result)


class MySink(discord.sinks.Sink):
    def __init__(self, guild):
        self.history = AudioHistory()
        self.anti_earrape = AntiEarrape(guild)
        self.seconds = 0

    def write(self, data, user):
        self.history.new_packet(data, user)
        self.anti_earrape.new_packet(data, user)


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

    loop = asyncio.get_event_loop()
    loop.create_task(message_sender_function())
    loop.create_task(muter_function())
    print("PRINT")


@bot.command()
async def record(ctx): 
    print("SAMPLE")
    voice = ctx.author.voice

    if not voice:
        await ctx.respond("You aren't in a voice channel!")

    vc = await voice.channel.connect()  # Connect to the voice channel the author is in.
    connections.update({ctx.guild.id: vc})  # Updating the cache with the guild and channel.

    vc.start_recording(
        MySink(ctx.guild),  # The sink type to use.
        lambda: None,  # What to do once done.
        ctx.channel  # The channel to disconnect from.
    )

# @bot.command()
# async def play(ctx: commands.Context, *, search: str):
#     """Plays a song.
#     If there are songs in the queue, this will be queued until the
#     other songs finished playing.
#     This command automatically searches from various sites if no URL is provided.
#     A list of these sites can be found here: https://rg3.github.io/youtube-dl/supportedsites.html
#     """

#     if not ctx.voice_state.voice:
#         await ctx.invoke(self._join)

#     async with ctx.typing():
#         try:
#             source = await YTDLSource.create_source(ctx, search, loop=self.bot.loop)
#         except YTDLError as e:
#             await ctx.send('An error occurred while processing this request: {}'.format(str(e)))
#         else:
#             song = Song(source)

#             await ctx.voice_state.songs.put(song)
#             await ctx.send('Enqueued {}'.format(str(source)))

bot.add_cog(Music(bot))

bot.run('OTg2NzE1MjY2MDUyNDAzMjMw.GnUKGD.9RnZnNhvtJ8eDcs-9hwJPl1iTip_tC85EIsLLI')
