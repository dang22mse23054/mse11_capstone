import {
	addDays,
	endOfDay,
	startOfDay,
	startOfMonth,
	endOfMonth,
	addMonths,
	startOfWeek,
	endOfWeek,
	// isSameDay,
	// differenceInCalendarDays,
} from 'date-fns';
import { createStaticRanges } from 'react-date-range';

const defineds = {
	startOfWeek: startOfWeek(new Date()),
	endOfWeek: endOfWeek(new Date()),
	startOfNextWeek: startOfWeek(addDays(new Date(), 7)),
	endOfNextWeek: endOfWeek(addDays(new Date(), 7)),
	startOfToday: startOfDay(new Date()),
	endOfToday: endOfDay(new Date()),
	startOfTomorow: startOfDay(addDays(new Date(), 1)),
	endOfTomorow: endOfDay(addDays(new Date(), 1)),
	startOfMonth: startOfMonth(new Date()),
	endOfMonth: endOfMonth(new Date()),
	startOfNextMonth: startOfMonth(addMonths(new Date(), 1)),
	endOfNextMonth: endOfMonth(addMonths(new Date(), 1)),
};


export const customStaticRanges = createStaticRanges([
	{
		label: '今日',
		range: () => ({
			startDate: defineds.startOfToday,
			endDate: defineds.endOfToday,
		}),
	},
	{
		label: '明日',
		range: () => ({
			startDate: defineds.startOfTomorow,
			endDate: defineds.endOfTomorow,
		}),
	},
	{
		label: '今週',
		range: () => ({
			startDate: defineds.startOfToday,
			endDate: defineds.endOfWeek,
		}),
	},
	{
		label: '来週',
		range: () => ({
			startDate: defineds.startOfNextWeek,
			endDate: defineds.endOfNextWeek,
		}),
	},
	{
		label: '今月',
		range: () => ({
			startDate: defineds.startOfToday,
			endDate: defineds.endOfMonth,
		}),
	},
	{
		label: '来月',
		range: () => ({
			startDate: defineds.startOfNextMonth,
			endDate: defineds.endOfNextMonth,
		}),
	},
]);